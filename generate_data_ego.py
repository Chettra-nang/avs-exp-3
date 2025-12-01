"""
Generate an ego-centric VLA dataset from highway-v0 with per-agent observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import trange

import highway_env  # noqa: F401

AMBULANCE_INDEX = 0


def classify_scene(env: gym.Env, agent_idx: int) -> str:
    vehicles = env.unwrapped.controlled_vehicles
    if agent_idx >= len(vehicles):
        return "Clear"

    agent = vehicles[agent_idx]
    ambulance = vehicles[AMBULANCE_INDEX]
    dx = ambulance.position[0] - agent.position[0]
    dy = ambulance.position[1] - agent.position[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    same_lane = agent.lane_index == ambulance.lane_index
    if dx < 0 and dist < 30.0 and same_lane:
        return "Yield"
    return "Clear"


def save_frame(frame: np.ndarray, path: Path) -> None:
    Image.fromarray(frame).save(path)


def generate_shard(
    shard_id: int,
    samples: int,
    output_dir: Path,
    seed: int,
) -> Path:
    import highway_env  # ensure env registration in subprocess
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(
        {
            "controlled_vehicles": 5,
            "vehicles_count": 10,
            "offscreen_rendering": True,
            "duration": 60,
        }
    )
    meta_path = output_dir / f"metadata_{shard_id}.jsonl"

    episode = shard_id
    with meta_path.open("w") as meta_f:
        for step in trange(samples, desc=f"shard-{shard_id}", leave=False):
            obs, info = env.reset(seed=seed + step)
            for agent_id, veh in enumerate(env.unwrapped.controlled_vehicles):
                env.unwrapped.observer_vehicle = veh
                frame = env.render()
                label = classify_scene(env, agent_id)
                file_name = f"img_{episode}_{step}_{agent_id}.jpg"
                save_frame(frame, output_dir / file_name)
                meta = {
                    "file_name": file_name,
                    "agent_id": agent_id,
                    "label": label,
                    "episode": episode,
                    "step": step,
                }
                meta_f.write(json.dumps(meta) + "\n")
    env.close()
    return meta_path


def merge_metadata(meta_paths, final_path: Path) -> None:
    with final_path.open("w") as out_f:
        for path in meta_paths:
            with path.open("r") as in_f:
                for line in in_f:
                    out_f.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ego-centric highway-env dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ego_vla"))
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_job = args.samples // args.n_jobs
    meta_paths = Parallel(n_jobs=args.n_jobs)(
        delayed(generate_shard)(
            shard_id=i,
            samples=per_job,
            output_dir=args.output_dir,
            seed=args.seed + i * 1000,
        )
        for i in range(args.n_jobs)
    )

    merge_metadata(meta_paths, args.output_dir / "metadata.jsonl")
    print(f"Ego dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
