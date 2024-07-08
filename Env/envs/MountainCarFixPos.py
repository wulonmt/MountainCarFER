import gymnasium as gym
from gymnasium.envs.classic_control import MountainCarEnv
from typing import Optional
import numpy as np
from gymnasium.envs.classic_control import utils

class MountainCarFixPos(MountainCarEnv):
    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0, init_x = 0):
        super().__init__(render_mode, goal_velocity)
        self.init_x = init_x

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        if options is not None:
            options.update({"low": -1, "high": -1})
        else:
            options = {"low": -1, "high": -1}
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}