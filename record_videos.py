"""
Director's cut video generation for the hybrid VLA agent.
Forces an ambulance behind the ego vehicle to showcase yielding behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from torchvision import transforms
from transformers import CLIPModel

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLASS_NAMES = ["Clear", "Hold", "Yield"]
VIDEO_LEN = 400
VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


class VLAClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = clip
        self.classifier = nn.Linear(self.clip.config.projection_dim, len(CLASS_NAMES))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        return self.classifier(image_features)


class VideoScenarioWrapper(gym.ObservationWrapper):
    """
    Forces a scenario with an ambulance behind the ego vehicle and appends VLA probs.
    """

    def __init__(self, env: gym.Env, classifier_path: Path, paint_ambulance: bool = False) -> None:
        super().__init__(env)
        self.paint_ambulance = paint_ambulance
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

        obs_shape = self.env.observation_space.shape
        self.base_dim = int(np.prod(obs_shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.base_dim + len(CLASS_NAMES),),
            dtype=np.float32,
        )
        self.last_probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)

    def _load_classifier(self, classifier_path: Path) -> nn.Module:
        model = VLAClassifier()
        checkpoint = torch.load(classifier_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        return model

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._force_ambulance()
        return self.observation(obs), info

    def observation(self, observation):
        if isinstance(observation, dict):
            obs_vec = np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in observation.values()])
        else:
            obs_vec = np.asarray(observation, dtype=np.float32).ravel()
        frame = self.env.render()
        if frame is not None:
            tensor = self.transform(Image.fromarray(frame).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.vla_model(tensor)
                self.last_probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return np.concatenate([obs_vec, self.last_probs], dtype=np.float32)

    def _force_ambulance(self) -> None:
        ego = self.env.unwrapped.vehicle
        vehicles = self.env.unwrapped.road.vehicles
        ambulance = next((v for v in vehicles if v is not ego), None)
        if ambulance:
            lane_index = ego.lane_index
            ambulance.lane_index = lane_index
            ambulance.position = ego.position + np.array([-15.0, 0.0])
            ambulance.speed = ego.speed + 2.0
            if self.paint_ambulance:
                ambulance.color = (255, 255, 0)


def make_directed_env(classifier_path: Path, paint_ambulance: bool = False) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make("highway-v0", render_mode="rgb_array")
        env.unwrapped.configure(
            {
                "observation": {"type": "Kinematics"},
                "vehicles_count": 20,
                "controlled_vehicles": 1,
                "lanes_count": 4,
                "duration": 40,
                "offscreen_rendering": True,
            }
        )
        env = VideoScenarioWrapper(env, classifier_path=classifier_path, paint_ambulance=paint_ambulance)
        return env

    return _init


def record_trained(classifier_path: Path, model_path: Path, paint_ambulance: bool = False) -> None:
    env_fn = make_directed_env(classifier_path, paint_ambulance=paint_ambulance)
    dummy_env = DummyVecEnv([env_fn])
    model = PPO.load(model_path, env=dummy_env)

    base_env = dummy_env.envs[0]

    def repaint_following_vehicle() -> None:
        raw_env = base_env.unwrapped
        ego = getattr(raw_env, "vehicle", None)
        road = getattr(raw_env, "road", None)
        if ego is None or road is None or not hasattr(road, "vehicles"):
            return
        for vehicle in road.vehicles:
            if vehicle is ego:
                continue
            dx = ego.position[0] - vehicle.position[0]
            if 5.0 <= dx <= 30.0:
                vehicle.color = (255, 255, 0)
                break

    video_env = VecVideoRecorder(
        dummy_env,
        str(VIDEO_DIR / "vla_agent"),
        record_video_trigger=lambda step: step == 0,
        video_length=VIDEO_LEN,
        name_prefix="vla-director-cut",
    )
    obs = video_env.reset()
    repaint_following_vehicle()
    for _ in range(VIDEO_LEN):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = video_env.step(action)
        repaint_following_vehicle()

    video_env.close()
    print("Saved video to", VIDEO_DIR / "vla_agent")


if __name__ == "__main__":
    classifier = Path("models/vla_classifier.pt")
    model_path = Path("models/hybrid_thesis_agent_run1.zip")
    record_trained(classifier, model_path, paint_ambulance=False)
