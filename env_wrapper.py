from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Tuple

import gymnasium as gym
import numpy as np


class AmbulanceYieldingEnv(gym.Wrapper):
    """
    Gymnasium wrapper that augments multi-agent observations with a small
    communication vector pointing to the ambulance.
    """

    def __init__(self, env: gym.Env, ambulance_index: int = 0) -> None:
        super().__init__(env)
        self.ambulance_index = ambulance_index

    def reset(self, **kwargs: Any) -> Tuple[MutableMapping[str, Any], Dict[str, Any]]:
        observations, info = self.env.reset(**kwargs)
        return self._append_messages(observations), info

    def step(
        self, actions: Mapping[str, Any]
    ) -> Tuple[MutableMapping[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        observations, reward, terminated, truncated, info = self.env.step(actions)
        return self._append_messages(observations), reward, terminated, truncated, info

    def _append_messages(self, observations: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        if not isinstance(observations, MutableMapping):
            raise TypeError("Expected a mapping of agent observations.")

        vehicles = getattr(self.env.unwrapped, "controlled_vehicles", None)
        if vehicles is None:
            raise AttributeError("The wrapped highway-env instance must expose 'controlled_vehicles'.")
        if len(vehicles) < len(observations):
            raise ValueError("Not enough controlled vehicles to map observations.")
        if self.ambulance_index >= len(vehicles):
            raise IndexError("Ambulance index is out of range for controlled vehicles.")

        positions = np.stack([np.asarray(v.position, dtype=np.float32) for v in vehicles])
        velocities = np.stack([np.asarray(v.velocity, dtype=np.float32) for v in vehicles])
        ambulance_position = positions[self.ambulance_index]
        ambulance_speed = np.float32(np.linalg.norm(velocities[self.ambulance_index]))

        relative_vectors = ambulance_position - positions[: len(observations)]
        speed_column = np.full((len(observations), 1), ambulance_speed, dtype=np.float32)
        messages = np.concatenate((relative_vectors, speed_column), axis=1).astype(np.float32)

        try:
            enhanced_observations = observations.__class__()  # preserves OrderedDict behavior
        except TypeError:
            enhanced_observations = {}
        for idx, (agent_id, agent_obs) in enumerate(observations.items()):
            agent_message = messages[idx]
            if isinstance(agent_obs, dict):
                augmented = dict(agent_obs)
                augmented["message"] = agent_message
            else:
                augmented = {"observation": agent_obs, "message": agent_message}
            enhanced_observations[agent_id] = augmented
        return enhanced_observations
