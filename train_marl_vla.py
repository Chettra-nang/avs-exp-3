"""
Multi-agent cooperative PPO training using CLIP-based visual cues appended to kinematics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torchvision import transforms
from transformers import CLIPModel

import highway_env  # noqa: F401

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLASS_NAMES = ["Clear", "Hold", "Yield"]
N_AGENTS = 5


class VLAClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = clip
        self.classifier = nn.Linear(self.clip.config.projection_dim, len(CLASS_NAMES))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        return self.classifier(image_features)


class MarlHybridWrapper(gym.Wrapper):
    """
    Augments each agent's kinematic observation with CLIP-derived ambulance probability,
    then flattens all agent observations/actions into a single vector for SB3.
    """

    def __init__(self, env: gym.Env, classifier_path: Path) -> None:
        super().__init__(env)
        self.device = torch.device("cpu")
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

        base_obs = self.env.observation_space
        if len(base_obs.shape) == 3:
            single_shape = base_obs.shape[1:]
            agent_count = base_obs.shape[0]
        else:
            single_shape = base_obs.shape
            agent_count = N_AGENTS
        if agent_count != N_AGENTS:
            raise ValueError("Unexpected number of agents in observation space.")

        self.single_dim = int(np.prod(single_shape)) + 1  # +1 for ambulance probability
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.single_dim * N_AGENTS,),
            dtype=np.float32,
        )

        base_action = self.env.action_space
        if isinstance(base_action, spaces.Tuple):
            self.action_space = spaces.MultiDiscrete([base_action.spaces[0].n] * N_AGENTS)
        elif isinstance(base_action, spaces.Discrete):
            self.action_space = spaces.MultiDiscrete([base_action.n] * N_AGENTS)
        else:
            raise ValueError("Unsupported action space.")

    def _load_classifier(self, classifier_path: Path) -> nn.Module:
        model = VLAClassifier()
        checkpoint = torch.load(classifier_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        return model

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        flat = self._augment_and_flatten(obs)
        return flat, info

    def step(self, action: Sequence[int]):
        action = np.asarray(action).reshape(N_AGENTS)
        obs, reward, terminated, truncated, info = self.env.step(tuple(int(a) for a in action))
        flat_obs = self._augment_and_flatten(obs)
        coop_reward = self._compute_cooperative_reward(reward)
        done = self._combine_flags(terminated)
        trunc = self._combine_flags(truncated)
        info = dict(info or {})
        info["coop_reward"] = coop_reward
        return flat_obs, coop_reward, done, trunc, info

    def _augment_and_flatten(self, obs) -> np.ndarray:
        kinematics = []
        for agent_obs in obs:
            kinematics.append(np.asarray(agent_obs, dtype=np.float32).ravel())

        frame = self.env.render()
        if frame is None:
            probs = np.zeros(N_AGENTS, dtype=np.float32)
        else:
            inputs = []
            for _ in range(N_AGENTS):
                inputs.append(self.transform(Image.fromarray(frame).convert("RGB")))
            batch = torch.stack(inputs).to(self.device)
            with torch.no_grad():
                logits = self.vla_model(batch)
                probs = torch.softmax(logits, dim=-1)[:, CLASS_NAMES.index("Yield")].cpu().numpy().astype(np.float32)

        augmented = []
        for kin, p in zip(kinematics, probs):
            augmented.append(np.concatenate([kin, [p]], dtype=np.float32))
        return np.concatenate(augmented, dtype=np.float32)

    def _compute_cooperative_reward(self, reward) -> float:
        if isinstance(reward, dict):
            base = np.mean(list(reward.values()))
        else:
            base = float(np.mean(reward))

        vehicles = getattr(self.env.unwrapped, "controlled_vehicles", [])
        speeds = [float(getattr(v, "speed", 0.0)) for v in vehicles]
        avg_speed = np.mean(speeds) if speeds else 0.0

        crashes = sum(1 for v in vehicles if getattr(v, "crashed", False))
        shared_reward = avg_speed - 10.0 * crashes

        if vehicles and speeds[0] > 25.0:
            shared_reward += 2.0  # ambulance bonus shared by all
        return shared_reward

    def _combine_flags(self, flags):
        if isinstance(flags, dict):
            return any(flags.values())
        return bool(flags)


def make_env(seed: int, rank: int, classifier_path: Path) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make("highway-v0", render_mode="rgb_array")
        env.unwrapped.configure(
            {
                "controlled_vehicles": N_AGENTS,
                "vehicles_count": 10,
                "ego_spacing": 2,
                "observation": {"type": "MultiAgentObservation", "observation_config": {"type": "Kinematics"}},
                "action": {"type": "MultiAgentAction", "action_config": {"type": "DiscreteMetaAction"}},
            }
        )
        env = MarlHybridWrapper(env, classifier_path=classifier_path)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MARL PPO with CLIP-based ambulance detection.")
    parser.add_argument("--classifier-path", type=Path, default=Path("models/vla_classifier.pt"))
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensorboard-log", type=Path, default=Path("marl_vla_tensorboard"))
    parser.add_argument("--output-path", type=Path, default=Path("models/marl_vla_agent.zip"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_fns = [make_env(args.seed, i, args.classifier_path) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.tensorboard_log.resolve()),
    )
    model.learn(total_timesteps=args.timesteps)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_path)
    vec_env.close()
    print(f"Saved MARL VLA agent to {args.output_path}")


if __name__ == "__main__":
    main()
