"""
Cooperative multi-agent PPO training on highway-v0 by flattening multi-agent observations/actions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env  # noqa: F401

AMBULANCE_INDEX = 0


class CooperativeWrapper(gym.Wrapper):
    """
    Converts the multi-agent highway-env interface into a single-agent one by flattening
    all agent observations and actions. Rewards are shared across agents to encourage cooperation.
    """

    def __init__(self, env: gym.Env, n_agents: int) -> None:
        super().__init__(env)
        self.n_agents = n_agents
        initial_obs, _ = self.env.reset()
        flat_obs = self._flatten_obs(initial_obs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=flat_obs.shape, dtype=np.float32
        )
        base_action = self._infer_action_space()
        self.action_space = spaces.MultiDiscrete([base_action] * self.n_agents)
        self.last_obs = flat_obs

    def _infer_action_space(self) -> int:
        action_space = getattr(self.env, "action_space", None)
        if isinstance(action_space, spaces.Discrete):
            return action_space.n
        if isinstance(action_space, spaces.Tuple):
            return action_space.spaces[0].n  # assume identical per agent
        raise ValueError("Unsupported action space for cooperative wrapper.")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        flat = self._flatten_obs(obs)
        self.last_obs = flat
        return flat, info

    def step(self, action: Union[np.ndarray, Sequence[int]]):
        multi_action = tuple(int(a) for a in np.asarray(action).reshape(self.n_agents))
        obs, reward, terminated, truncated, info = self.env.step(multi_action)
        flat_obs = self._flatten_obs(obs)
        global_reward = self._shared_reward(reward)
        done = self._any_flag(terminated)
        trunc = self._any_flag(truncated)
        shared_info = {"raw_info": info, "global_reward": global_reward}
        self.last_obs = flat_obs
        return flat_obs, global_reward, done, trunc, shared_info

    def _any_flag(self, flag) -> bool:
        if isinstance(flag, dict):
            return any(flag.values())
        return bool(flag)

    def _flatten_obs(self, obs) -> np.ndarray:
        agent_obs = []
        if isinstance(obs, dict):
            for val in obs.values():
                agent_obs.append(np.asarray(val, dtype=np.float32).ravel())
        else:
            for item in obs:
                agent_obs.append(np.asarray(item, dtype=np.float32).ravel())
        return np.concatenate(agent_obs, dtype=np.float32)

    def _shared_reward(self, reward) -> float:
        if isinstance(reward, dict):
            base = sum(float(r) for r in reward.values())
        else:
            base = float(np.sum(reward))

        vehicles = getattr(self.env.unwrapped, "controlled_vehicles", [])
        bonus = 0.0
        if vehicles:
            ambulance_speed = float(getattr(vehicles[AMBULANCE_INDEX], "speed", 0.0))
            if ambulance_speed >= 30.0:
                bonus += 1.5 * self.n_agents  # shared bonus when ambulance cruises fast
        crash_penalty = 0.0
        if any(getattr(v, "crashed", False) for v in vehicles):
            crash_penalty -= 3.0 * self.n_agents  # shared pain
        return base + bonus + crash_penalty


def make_env(seed: int, rank: int) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make("highway-v0")
        env.configure(
            {
                "controlled_vehicles": 5,
                "vehicles_count": 10,
                "ego_spacing": 2,
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "Kinematics"},
                },
                "action": {"type": "MultiAgentAction", "action_config": {"type": "DiscreteMetaAction"}},
                "duration": 40,
            }
        )
        env = CooperativeWrapper(env, n_agents=5)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cooperative MARL training on highway-env.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=Path("models/marl_cooperative.zip"))
    parser.add_argument("--tensorboard-log", type=Path, default=Path("marl_tensorboard"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_fns = [make_env(args.seed, i) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=str(args.tensorboard_log.resolve()),
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_path)
    vec_env.close()
    print(f"Saved cooperative MARL model to {args.output_path}")


if __name__ == "__main__":
    main()
