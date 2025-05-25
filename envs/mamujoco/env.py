import random
import numpy as np
from envs.mamujoco.env_wrappers import ShareDummyVecEnv
from envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti


def make_env(config):
    def get_env_fn(rank):
        def init_env():
            if config["env_name"] == "mujoco":
                env_args = {
                    "scenario": config["scenario"],
                    "agent_conf": config["agent_conf"],
                    "agent_obsk": config["agent_obsk"],
                    "episode_limit": 1000
                }
                env = MujocoMulti(env_args=env_args)
            else:
                print("Can not support the " + config["env_name"] + "environment.")
                raise NotImplementedError
            env.seed(config["seed"])
            return env
        return init_env
    return ShareDummyVecEnv([get_env_fn(0)])


class MaMujocoWrapper:
    
    def __init__(self, map_name, seed):
        random.seed(seed)
        np.random.seed(seed)
        if map_name == "Ant-v2":
            agent_conf = "2x4"
        elif map_name == "HalfCheetah-v2":
            agent_conf = "6x1"
        elif map_name == "Hopper-v2":
            agent_conf = "3x1"
        else:
            raise NotImplementedError
        self.config = {
            "env_name": "mujoco",
            "scenario": map_name,
            "agent_conf": agent_conf,
            "agent_obsk": 1,
            "seed": seed
        }
        self.env = make_env(self.config)
    
    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)
    
    def close(self):
        self.env.close()