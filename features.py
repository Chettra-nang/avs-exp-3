from typing import Tuple

import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DINOv2FeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that wraps the DINOv2 ViT-S/14 backbone while keeping
    everything frozen to minimize VRAM usage.
    """

    def __init__(self, observation_space: spaces.Box) -> None:
        if len(observation_space.shape) != 3:
            raise ValueError(
                "Expected image observations with shape (C, H, W). "
                f"Got {observation_space.shape} instead."
            )

        self.c, self.h, self.w = observation_space.shape

        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        features_dim = getattr(model, "embed_dim", None)
        if features_dim is None:
            raise AttributeError("Could not determine DINOv2 embedding dimension.")

        super().__init__(observation_space, features_dim=features_dim)
        self.backbone = model

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype != torch.float32:
            observations = observations.float()
        resized = F.interpolate(
            observations,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        resized = torch.clamp(resized / 255.0, 0.0, 1.0)
        features = self.backbone(resized)
        return features.view(features.size(0), -1)
