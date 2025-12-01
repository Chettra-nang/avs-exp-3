"""
Train a PPO agent on highway-v0 using a hybrid observation that fuses kinematics with
visual embeddings from a pre-trained VLA classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torchvision import transforms
from transformers import CLIPModel

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLASS_NAMES = ["Clear", "Hold", "Yield"]
CONTROLLED_VEHICLES = 4


class VLAClassifier(nn.Module):
    """Matches the architecture used during CLIP fine-tuning."""

    def __init__(self) -> None:
        super().__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = clip
        self.classifier = nn.Linear(self.clip.config.projection_dim, len(CLASS_NAMES))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        return self.classifier(image_features)


class HybridObservationWrapper(gym.ObservationWrapper):
    """
    Appends VLA classifier probabilities to the flattened kinematics observation and
    applies reward shaping based on predicted yields.
    """

    def __init__(self, env: gym.Env, classifier_path: Path) -> None:
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
        self.vla_model.eval()
        for param in self.vla_model.parameters():
            param.requires_grad = False

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

    def _load_classifier(self, classifier_path: Path) -> nn.Module:
        model = VLAClassifier()
        checkpoint = torch.load(classifier_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
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
        hybrid_obs = np.concatenate([obs_vec, probs], dtype=np.float32)
        return hybrid_obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        hybrid_obs = self.observation(obs)
        shaped = self._shape_reward()
        reward += shaped
        return hybrid_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._paint_ambulance()
        self.prev_lane_id = self._get_lane_id()
        return self.observation(obs), info

    def _paint_ambulance(self) -> None:
        vehicles = getattr(self.env.unwrapped, "controlled_vehicles", None)
        if vehicles and len(vehicles) > 0:
            vehicles[0].color = (255, 255, 0)

    def _compute_visual_probs(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            return self.last_probs
        image = Image.fromarray(frame).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.vla_model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
        return probs

    def _shape_reward(self) -> float:
        lane_id = self._get_lane_id()
        reward = 0.0
        yield_prob = float(self.last_probs[CLASS_NAMES.index("Yield")])
        if yield_prob > 0.7 and lane_id is not None:
            if lane_id == 0:
                reward -= 1.0
            elif self.prev_lane_id == 0 and lane_id > self.prev_lane_id:
                reward += 1.0
        self.prev_lane_id = lane_id
        return reward

    def _get_lane_id(self) -> int | None:
        vehicle = getattr(self.env, "vehicle", None)
        if vehicle is None:
            return None
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is None:
            return None
        if isinstance(lane_index, (tuple, list)) and len(lane_index) >= 3:
            return int(lane_index[2])
        try:
            return int(lane_index)
        except (TypeError, ValueError):
            return None


def make_env(rank: int, seed: int, classifier_path: Path) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        import highway_env  # ensure envs register in each subprocess
        env = gym.make("highway-v0", render_mode="rgb_array")
        env.unwrapped.configure(
            {
                "observation": {"type": "Kinematics"},
                "vehicles_count": 20,
                "controlled_vehicles": CONTROLLED_VEHICLES,
                "lanes_count": 4,
                "duration": 40,
                "action": {"type": "DiscreteMetaAction"},
                "offscreen_rendering": True,
            }
        )
        env = HybridObservationWrapper(env, classifier_path=classifier_path)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid PPO agent with VLA visual cues.")
    parser.add_argument("--classifier-path", type=Path, default=Path("models/vla_classifier.pt"))
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=Path("models/hybrid_ppo_agent.zip"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vec_env = SubprocVecEnv(
        [make_env(i, args.seed, classifier_path=args.classifier_path) for i in range(args.n_envs)],
        start_method="spawn",
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(Path("./ppo_vla_tensorboard").resolve()),
    )
    model.learn(total_timesteps=args.timesteps)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_path)
    vec_env.close()
    print(f"Saved hybrid PPO agent to {args.output_path}")


if __name__ == "__main__":
    main()
