from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from envs.smacv2.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from envs.utils import silence_stderr


class StarCraft2Env(StarCraftCapabilityEnvWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_agents = self.env.n_agents
        self.n_enemies = self.env.n_enemies

    def reset(self):
        with silence_stderr():
            super().reset()
        obs = self.get_obs()
        states = self.get_state()
        avails = self.get_avail_actions()
        return obs, states, avails
    
    def step(self, actions):
        with silence_stderr():
            rewards, dones, infos = super().step(actions)
        myinfo = {}
        if dones:
            myinfo["dead_allies"] = infos.get("dead_allies", 0) / self.n_agents
            myinfo["dead_enemies"] = infos.get("dead_enemies", 0) / self.n_enemies
            myinfo["go_count"] = self.env._episode_steps
            myinfo["won"] = infos.get("battle_won", False)
            myinfo = {0: myinfo}
        states = self.env.get_state()
        obs = self.env.get_obs()
        avails = self.env.get_avail_actions()
        return obs, states, rewards, dones, myinfo, avails


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                with silence_stderr():
                    env = StarCraft2Env(**all_args.config)
                env_info = env.get_env_info()
                env.observation_space = env_info["obs_shape"]
                env.share_observation_space = env_info["state_shape"]
                env.action_space = env_info["n_actions"]
                env.reset()
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


class SMACEnv:
    
    def __init__(self, config):
        self.real_env = make_eval_env(config)

    def reset(self):
        obs, state, avail = self.real_env.reset()
        return obs, state, avail
    
    def step(self, actions):
        obs, states, rewards, dones, infos, avails = self.real_env.step([actions])
        infos = infos[0]
        return obs, states, rewards, dones, infos, avails