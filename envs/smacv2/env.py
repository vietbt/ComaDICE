import random
import numpy as np
from envs.utils import silence_stderr


class Config:
    
    n_eval_rollout_threads = 1
    env_name = "StarCraft2"

    def __init__(self, map_name, seed):
        self.map_name = map_name
        self.seed = seed
        self.config = self.read_smac_config(map_name)
    
    def read_smac_config(self, map_name):
        map_type, params = map_name.lower().split("_", 1)
        if map_type not in ["protoss", "terran", "zerg"]:
            raise
        n_agents, _, n_enemy = params.split("_")
        import yaml
        with open(f"envs/smacv2/configs/sc2_gen_{map_type}.yaml", "r") as f:
            config = yaml.safe_load(f)["env_args"]
            config["capability_config"]["n_units"] = int(n_agents)
            config["capability_config"]["n_enemies"] = int(n_enemy)
        return config


class SMACWrapper:

    def __init__(self, env_name, seed=0):
        self.init(env_name)
        self.set_seed(seed)
    
    def init(self, env_name):
        self.close()
        from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper as StarCraft2Env
        with silence_stderr():
            self.env = StarCraft2Env(**self.read_smac_config(env_name))
            self.env_info = self.env.get_env_info()
        self.st_dim = self.env_info["state_shape"]
        self.ob_dim = self.env_info["obs_shape"]
        self.ac_dim = self.env_info["n_actions"]
        self.n_agents = self.env.env.n_agents
        self.n_enemies = self.env.env.n_enemies
        self.max_len = self.env_info["episode_limit"]
        self.env_name = env_name

    def read_smac_config(self, map_name):
        map_type, params = map_name.lower().split("_", 1)
        if map_type not in ["protoss", "terran", "zerg"]:
            raise
        n_agents, _, n_enemy = params.split("_")
        import yaml
        with open(f"envs/smacv2/configs/sc2_gen_{map_type}.yaml", "r") as f:
            config = yaml.safe_load(f)["env_args"]
            config["capability_config"]["n_units"] = int(n_agents)
            config["capability_config"]["n_enemies"] = int(n_enemy)
        return config
    
    def set_seed(self, seed):
        self.env.env._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
    
    def get_env_specs(self):
        return self.ob_dim, self.ac_dim, self.n_agents
    
    def reset(self):
        try:
            with silence_stderr():
                self.env.reset()
            self.set_seed(self.seed + 1)
            state = self.env.get_state()
            obs = self.env.get_obs()
            avails = self.env.get_avail_actions()
            return obs, state, avails
        except Exception as e:
            print(e)
            self.init(self.env_name)
            return self.reset()

    def close(self):
        try:
            self.env.close()
            del self.env
        except Exception as e:
            pass
    
    def step(self, actions, reset_after_done=False):
        with silence_stderr():
            reward, done, info = self.env.step(actions)
        myinfo = {}
        if done:
            myinfo["dead_allies"] = info.get("dead_allies", 0) / self.n_agents
            myinfo["dead_enemies"] = info.get("dead_enemies", 0) / self.n_enemies
            myinfo["go_count"] = self.env._episode_steps
            myinfo["won"] = info.get("battle_won", False)
            if reset_after_done:
                self.reset()
            myinfo = {0: myinfo}
        state = self.env.get_state()
        obs = self.env.get_obs()
        avails = self.env.get_avail_actions()
        return obs, state, reward, done, myinfo, avails