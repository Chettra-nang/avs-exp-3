"""
VERIFIED Synthetic VLA dataset generator for highway-env.
Uses multi-agent observation/action wrappers, per-agent labeling, and forced interactions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import gymnasium as gym
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

# Registers highway-env environments with Gymnasium
import highway_env  # noqa: F401

# Environment settings (highway-env v1.8+)
ENV_ID = "highway-v0"
AMBULANCE_INDEX = 0  # ego ambulance
LEARNER_COUNT = 3  # number of learning agents (non-ambulance)
NPC_MIN = 2
NPC_MAX = 8
INTERACTION_DISTANCE = 45.0  # meters threshold to consider an interaction
NO_INTERACTION_LIMIT = 25  # steps without interaction before forcing one
MAX_EPISODE_STEPS = 300
FRAME_SIZE = (256, 256)

CLASSES = ("Yield", "Hold", "Clear")


def _ensure_env_registered() -> None:
    """Ensure the target env id is registered; provide a clear hint if not."""
    # Re-import defensively to register inside worker processes.
    import highway_env  # noqa: F401
    try:
        gym.spec(ENV_ID)
    except gym.error.Error as exc:  # gymnasium re-exports gym.error
        raise RuntimeError(
            f"Environment id '{ENV_ID}' is not registered. "
            "Install/upgrade highway-env (>=1.8) and ensure it is importable before gym.make."
        ) from exc


def make_env(seed: int) -> gym.Env:
    """Create a multi-agent highway env with proper observation/action wrappers."""
    _ensure_env_registered()
    env = gym.make(ENV_ID, render_mode="rgb_array")
    config: Dict = {
        "controlled_vehicles": 1 + LEARNER_COUNT,
        "vehicles_count": 1 + LEARNER_COUNT + NPC_MIN,
        "screen_width": FRAME_SIZE[0],
        "screen_height": FRAME_SIZE[1],
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["x", "y", "vx", "vy"],
                "absolute": False,
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {"type": "DiscreteMetaAction"},
        },
        "show_trajectories": False,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
    }
    env.unwrapped.configure(config)
    env.reset(seed=seed)
    _paint_ambulance(env)
    return env


def _paint_ambulance(env: gym.Env) -> None:
    vehicles = getattr(env.unwrapped, "controlled_vehicles", None)
    if vehicles and len(vehicles) > AMBULANCE_INDEX:
        vehicles[AMBULANCE_INDEX].color = (255, 255, 0)


def classify_scene(env: gym.Env, agent_idx: int) -> Tuple[str, float, float, int]:
    """
    Classify a specific learner relative to the ambulance.
    Returns (label, distance, lateral_diff, lane_id).
    """
    vehicles = env.unwrapped.controlled_vehicles
    if len(vehicles) <= agent_idx:
        return "Clear", 999.0, 999.0, -1

    ambulance = vehicles[AMBULANCE_INDEX]
    agent = vehicles[agent_idx]

    dx = ambulance.position[0] - agent.position[0]
    dy = ambulance.position[1] - agent.position[1]
    dist = float(np.sqrt(dx ** 2 + dy ** 2))

    lane_index = agent.lane_index
    lane_id = lane_index[2] if isinstance(lane_index, (list, tuple)) and len(lane_index) >= 3 else -1

    label = "Clear"
    if dx < 0 and dist < INTERACTION_DISTANCE:
        label = "Yield" if abs(dy) < 2.5 else "Hold"

    return label, dist, float(dy), int(lane_id)


def force_interaction(env: gym.Env, rng: np.random.Generator, gap: float = 20.0) -> None:
    """Teleport the ambulance behind a random learner to guarantee interactions without crashes."""
    vehicles = env.unwrapped.controlled_vehicles
    if len(vehicles) < 2:
        return

    ambulance = vehicles[AMBULANCE_INDEX]
    target_idx = int(rng.integers(1, len(vehicles)))
    target = vehicles[target_idx]

    lane = env.unwrapped.road.network.get_lane(target.lane_index)
    s, r = lane.local_coordinates(target.position)
    new_s = max(0.0, s - gap - float(rng.uniform(0, 5.0)))

    ambulance.position = np.asarray(lane.position(new_s, r), dtype=np.float32)
    ambulance.speed = float(target.speed + 2.0)
    ambulance.heading = float(target.heading)
    ambulance.lane_index = target.lane_index
    ambulance.lane = lane
    ambulance.crashed = False
    target.crashed = False


def ensure_dirs(root: Path, split: str) -> None:
    for cls in CLASSES:
        (root / split / cls).mkdir(parents=True, exist_ok=True)


def save_frame(frame: np.ndarray, path: Path) -> None:
    Image.fromarray(frame).save(path)


def start_episode(env: gym.Env, rng: np.random.Generator) -> None:
    npc_count = int(rng.integers(NPC_MIN, NPC_MAX + 1))
    env.unwrapped.config["vehicles_count"] = 1 + LEARNER_COUNT + npc_count
    env.reset(seed=int(rng.integers(0, 1_000_000_000)))
    _paint_ambulance(env)


def generate_shard(
    worker_id: int,
    start_idx: int,
    num_samples: int,
    out_dir: Path,
    split: str,
    seed: int,
) -> Path:
    rng = np.random.default_rng(seed + worker_id)
    env = make_env(seed + worker_id)
    meta_path = out_dir / f"metadata_worker_{worker_id}.jsonl"

    steps_since_interaction = 0
    episode_steps = 0

    with meta_path.open("w") as meta_f:
        pbar = tqdm(total=num_samples, desc=f"worker-{worker_id}", position=worker_id, leave=False)
        collected = 0
        start_episode(env, rng)

        while collected < num_samples:
            if episode_steps >= MAX_EPISODE_STEPS:
                start_episode(env, rng)
                episode_steps = 0

            actions = tuple(env.action_space.sample())  # Multi-agent discrete actions
            _, _, terminated, truncated, _ = env.step(actions)
            frame = env.render()
            if frame is None:
                raise RuntimeError("Env render() returned None. Ensure render_mode='rgb_array'.")

            # Interaction check across learners
            min_dist = INTERACTION_DISTANCE + 1.0
            for idx in range(1, len(env.unwrapped.controlled_vehicles)):
                _, dist, _, _ = classify_scene(env, idx)
                min_dist = min(min_dist, dist)

            if min_dist < INTERACTION_DISTANCE:
                steps_since_interaction = 0
            else:
                steps_since_interaction += 1
                if steps_since_interaction >= NO_INTERACTION_LIMIT:
                    force_interaction(env, rng)
                    steps_since_interaction = 0

            for idx in range(1, len(env.unwrapped.controlled_vehicles)):
                if collected >= num_samples:
                    break
                agent = env.unwrapped.controlled_vehicles[idx]
                if getattr(agent, "crashed", False):
                    continue
                label, dist, dy, lane_id = classify_scene(env, idx)
                if label == "Clear" and rng.random() > 0.05:
                    continue

                sample_id = start_idx + collected
                file_rel = f"{split}/{label}/w{worker_id}_{sample_id:06d}.png"
                save_frame(frame, out_dir / file_rel)

                meta = {
                    "file_name": file_rel,
                    "text": label,
                    "ground_truth_dist": float(dist),
                    "ground_truth_lane": lane_id,
                    "ground_truth_lane_diff": float(dy),
                    "agent_id": idx,
                }
                meta_f.write(json.dumps(meta) + "\n")
                collected += 1
                pbar.update(1)

            episode_steps += 1
            if terminated or truncated:
                start_episode(env, rng)
                episode_steps = 0

    env.close()
    return meta_path


def merge_metadata(meta_paths: Iterable[Path], final_path: Path) -> None:
    with final_path.open("w") as out_f:
        for path in meta_paths:
            with path.open("r") as in_f:
                for line in in_f:
                    out_f.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VLA highway-env dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/vla_highway"), help="Root directory for dataset.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name.")
    parser.add_argument("--samples", type=int, default=50_000, help="Number of (image, text) pairs to generate.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Parallel workers.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs(out_dir, args.split)

    per_worker = math.ceil(args.samples / args.n_jobs)
    shards: List[Tuple[int, int]] = []
    for w in range(args.n_jobs):
        start = w * per_worker
        count = min(per_worker, args.samples - start)
        if count > 0:
            shards.append((start, count))

    meta_paths = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(generate_shard)(
            worker_id=w,
            start_idx=start,
            num_samples=count,
            out_dir=out_dir,
            split=args.split,
            seed=args.seed,
        )
        for w, (start, count) in enumerate(shards)
    )

    final_meta = out_dir / "metadata.jsonl"
    merge_metadata(meta_paths, final_meta)

    for path in meta_paths:
        path.unlink(missing_ok=True)

    print(f"Dataset written to {out_dir} with metadata at {final_meta}")


if __name__ == "__main__":
    main()
