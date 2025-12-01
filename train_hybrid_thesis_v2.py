"""
Hybrid PPO training with thesis-grade logging for yield compliance and collisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torchvision import transforms
from transformers import CLIPModel

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLASS_NAMES = ["Clear", "Hold", "Yield"]

class VLAClassifier(nn.Module):
    """Matches the fine-tuned CLIP classifier architecture."""
    def __init__(self) -> None:
        super().__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = clip
        self.classifier = nn.Linear(self.clip.config.projection_dim, len(CLASS_NAMES))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        return self.classifier(image_features)

class ThesisHybridObservationWrapper(gym.ObservationWrapper):
    """
    Extends the hybrid observation with per-step logging info for thesis metrics.
    """
    def __init__(self, env: gym.Env, classifier_path: Path, paint_ambulance: bool = True) -> None:
        super().__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
        self.vla_model = self._load_classifier(classifier_path)
        for p in self.vla_model.parameters():
            p.requires_grad = False
        self.vla_model.eval()
        self.paint_ambulance = paint_ambulance

        obs_shape = self.env.observation_space.shape
        self.base_dim = int(np.prod(obs_shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.base_dim + len(CLASS_NAMES),),
            dtype=np.float32,
        )

        self.last_probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        self.prev_lane_id = None
        self.pending_yield = False

    def _load_classifier(self, classifier_path: Path) -> nn.Module:
        model = VLAClassifier()
        checkpoint = torch.load(classifier_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        return model

    def observation(self, observation):
        if isinstance(observation, dict):
            obs_vec = np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in observation.values()])
        else:
            obs_vec = np.asarray(observation, dtype=np.float32).ravel()

        probs = self._compute_visual_probs()
        self.last_probs = probs
        return np.concatenate([obs_vec, probs], dtype=np.float32)

    def step(self, action):
        prev_lane = self.prev_lane_id
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        hybrid_obs = self.observation(obs)
        lane_id = self._get_lane_id()
        yield_prob = float(self.last_probs[CLASS_NAMES.index("Yield")])
        yield_needed = bool(yield_prob > 0.7 and lane_id == 0)
        if yield_needed:
            self.pending_yield = True

        yield_success = False
        if self.pending_yield and lane_id is not None and prev_lane is not None and lane_id > prev_lane:
            yield_success = True
            self.pending_yield = False

        shaped = self._shape_reward(lane_id, yield_prob, prev_lane)
        reward += shaped

        info["yield_needed"] = yield_needed
        info["yield_success"] = yield_success
        info.setdefault("crashed", getattr(self.env.unwrapped.vehicle, "crashed", False))
        info.setdefault("speed", float(getattr(self.env.unwrapped.vehicle, "speed", 0.0)))

        self.prev_lane_id = lane_id
        return hybrid_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.paint_ambulance:
            self._paint_ambulance()
        self.prev_lane_id = self._get_lane_id()
        self.pending_yield = False
        return self.observation(obs), info

    def _compute_visual_probs(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            return self.last_probs
        tensor = self.transform(Image.fromarray(frame).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.vla_model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
        return probs

    def _shape_reward(self, lane_id: int | None, yield_prob: float, prev_lane: int | None) -> float:
        reward = 0.0
        speed = float(getattr(self.env.unwrapped.vehicle, "speed", 0.0))
        if yield_prob > 0.7 and lane_id is not None:
            if lane_id == 0:
                reward -= 1.0
            elif prev_lane == 0 and lane_id > prev_lane:
                reward += 1.0
        if speed < 20.0:
            reward -= 0.5
        return reward

    def _get_lane_id(self) -> int | None:
        vehicle = getattr(self.env.unwrapped, "vehicle", None)
        if vehicle is None:
            return None
        lane_index = getattr(vehicle, "lane_index", None)
        if isinstance(lane_index, (tuple, list)) and len(lane_index) >= 3:
            return int(lane_index[2])
        try:
            return int(lane_index)
        except (TypeError, ValueError):
            return None

    def _paint_ambulance(self) -> None:
        vehicles = getattr(self.env.unwrapped, "controlled_vehicles", None)
        if vehicles and len(vehicles) > 0:
            vehicles[0].color = (255, 255, 0)

class ThesisMetricsCallback(BaseCallback):
    """Logs collision rates and yield compliance statistics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.crash_count = 0
        self.episode_count = 0
        self.yield_events = 0
        self.yield_compliances = 0
        self.speed_sum = 0.0
        self.speed_samples = 0

    def _on_step(self) -> bool:
        infos: List[dict] = self.locals.get("infos", [])
        for info in infos:
            if info is None:
                continue
            if info.get("crashed", False):
                self.crash_count += 1
            if info.get("yield_needed", False):
                self.yield_events += 1
            if info.get("yield_success", False):
                self.yield_compliances += 1
            if "speed" in info:
                self.speed_sum += float(info["speed"])
                self.speed_samples += 1
            if "episode" in info:
                self.episode_count += 1
        return True

    def _on_rollout_end(self) -> None:
        episodes = max(self.episode_count, 1)
        events = max(self.yield_events, 1)
        speeds = max(self.speed_samples, 1)

        collision_rate = self.crash_count / episodes
        yield_compliance = self.yield_compliances / events
        avg_speed = self.speed_sum / speeds

        self.logger.record("thesis/collision_rate", collision_rate)
        self.logger.record("thesis/yield_compliance", yield_compliance)
        self.logger.record("thesis/avg_speed", avg_speed)

        if self.verbose > 0:
            print(
                f"[ThesisMetrics] Collision Rate: {collision_rate:.3f}, "
                f"Yield Compliance: {yield_compliance:.3f}, Avg Speed: {avg_speed:.2f}"
            )

        self.reset_stats()

def make_env(rank: int, seed: int, classifier_path: Path) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        import highway_env  # ensure registration

        env = gym.make("highway-v0", render_mode="rgb_array")
        env.unwrapped.configure(
            {
                "observation": {"type": "Kinematics"},
                "vehicles_count": 20,
                "controlled_vehicles": 1,
                "lanes_count": 4,
                "duration": 40,
                "action": {"type": "DiscreteMetaAction"},
                "offscreen_rendering": True,
            }
        )
        env = ThesisHybridObservationWrapper(env, classifier_path=classifier_path)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid PPO agent with thesis-grade logging.")
    parser.add_argument("--classifier-path", type=Path, default=Path("models/multi/vla_classifier.pt"))
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=Path("models/multi/hybrid_thesis_agent.zip"))
    parser.add_argument("--tensorboard-log", type=Path, default=Path("ppo_vla_thesis_logs"))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    vec_env = SubprocVecEnv(
        [make_env(i, args.seed, classifier_path=args.classifier_path) for i in range(args.n_envs)],
        start_method="spawn",
    )

    callback = ThesisMetricsCallback(verbose=1)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.tensorboard_log.resolve()),
    )
    model.learn(total_timesteps=args.timesteps, callback=callback)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_path)
    vec_env.close()
    print(f"Saved hybrid thesis PPO agent to {args.output_path}")

if __name__ == "__main__":
    main()
