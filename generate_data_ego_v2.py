"""
Generate ego-centric dataset with camera centered on each controlled vehicle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from tqdm import tqdm

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
    dist = float(np.hypot(dx, dy))

    if dx < 0 and dist < 30.0 and abs(dy) < 2.0:
        return "Yield"
    if abs(dy) > 2.0 and dist < 20.0:
        return "Hold"
    return "Clear"


def capture_agent_frame(env: gym.Env, agent) -> np.ndarray:
    env.unwrapped.observer_vehicle = agent
    viewer = getattr(env.unwrapped, "viewer", None)
    if viewer is not None:
        viewer.observer_vehicle = agent
    env.unwrapped.config["centering_position"] = [0.3, 0.5]
    frame = env.render()
    if frame is None:
        raise RuntimeError("Env render returned None; ensure offscreen rendering is enabled.")
    return frame


def save_frame(frame: np.ndarray, path: Path) -> None:
    Image.fromarray(frame).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ego-centric dataset with centered camera.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ego_vla_v2"))
    parser.add_argument("--samples", type=int, default=10_000, help="Total number of agent frames to save.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "metadata.jsonl"

    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure(
        {
            "controlled_vehicles": 5,
            "vehicles_count": 10,
            "ego_spacing": 2,
            "offscreen_rendering": True,
            "screen_width": 600,
            "screen_height": 300,
            "scaling": 5.5,
            "centering_position": [0.3, 0.5],
        }
    )

    saved = 0
    step_idx = 0
    with meta_path.open("w") as meta_f:
        obs, info = env.reset(seed=args.seed)
        while saved < args.samples:
            actions = env.action_space.sample()
            if step_idx == 0:
                obs, info = env.reset(seed=args.seed)
                terminated = False
                truncated = False
            else:
                obs, reward, terminated, truncated, info = env.step(actions)

            vehicles = env.unwrapped.controlled_vehicles
            if step_idx % 10 == 0 and len(vehicles) > 0:
                target_idx = np.random.randint(len(vehicles))
                target = vehicles[target_idx]
                ambulance = vehicles[AMBULANCE_INDEX]
                ambulance.position = target.position + np.array([-15.0, 0.0])
                ambulance.speed = target.speed + 5.0
                ambulance.lane_index = target.lane_index

            vehicles = env.unwrapped.controlled_vehicles
            if vehicles:
                vehicles[0].color = (255, 255, 0)
            for agent_id, agent in enumerate(vehicles):
                if saved >= args.samples:
                    break
                frame = capture_agent_frame(env, agent)
                label = classify_scene(env, agent_id)
                if label == "Clear" and np.random.random() > 0.1:
                    continue
                file_name = f"agent_{agent_id}_step_{step_idx:06d}.jpg"
                save_frame(frame, args.output_dir / file_name)
                meta = {"file_name": file_name, "agent_id": agent_id, "label": label, "step": step_idx}
                meta_f.write(json.dumps(meta) + "\n")
                saved += 1

            step_idx += 1
            if terminated or truncated:
                obs, info = env.reset()

    env.close()
    print(f"Ego-centric dataset saved to {args.output_dir} ({saved} samples).")


if __name__ == "__main__":
    main()
