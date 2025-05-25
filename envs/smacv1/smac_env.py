from envs.smacv1.StarCraft2_Env import StarCraft2Env
from envs.smacv1.smac_maps import get_map_params
from envs.smacv1.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
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