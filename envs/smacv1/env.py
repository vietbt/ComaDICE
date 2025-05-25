import random
import numpy as np
from envs.utils import silence_stderr
from envs.smacv1.StarCraft2_Env import StarCraft2Env as SMACEnv


class Config:

    add_local_obs = False
    add_move_state = False
    add_visible_state = False
    add_distance_state = False
    add_xy_state = False
    add_enemy_action_state = False
    add_agent_id = True
    use_state_agent = True
    use_mustalive = True
    add_center_xy = True
    use_stacked_frames = False
    stacked_frames = 1
    use_obs_instead_of_state = False
    n_eval_rollout_threads = 1
    env_name = "StarCraft2"

    def __init__(self, map_name, seed):
        self.map_name = map_name
        self.seed = seed


class SMACWrapper:
    
    def __init__(self, map_name, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        self.config = Config(map_name, seed)
        with silence_stderr():
            self.env = SMACEnv(self.config)
            self.env.seed(seed)
    
    def reset(self):
        with silence_stderr():
            return self.env.reset()

    def step(self, actions):
        with silence_stderr():
            return self.env.step(actions)
    
    def close(self):
        with silence_stderr():
            self.env.close()