import os
import time
from concurrent.futures import ThreadPoolExecutor as Pool


EPOCHS = {
    "2c_vs_64zg": 40,
    "6h_vs_8z": 400,
    "5m_vs_6m": 10,
    "corridor": 20,
}

def run_cmd(env_name, mode, seed, n_epochs=None):
    if n_epochs is None:
        n_epochs = EPOCHS[env_name]
    cmd = f"python -u main.py --env_name {env_name} --mode {mode} --seed {seed} --n_epochs {n_epochs}"
    print(cmd)
    # os.system(cmd)


def run_mamujoco(p, seed):
    tasks = []
    for env_name in ["Hopper-v2", "Ant-v2", "HalfCheetah-v2"]:
        for mode in ["expert", "medium", "medium-replay", "medium-expert"]:
            tasks.append(p.submit(run_cmd, env_name, mode, seed, 25))
            time.sleep(0.1)
    return tasks


def run_smacv1(p, seed):
    tasks = []
    for env_name in ["5m_vs_6m", "2c_vs_64zg", "6h_vs_8z", "corridor"]:
        for mode in ["good", "medium", "poor"]:
            tasks.append(p.submit(run_cmd, env_name, mode, seed))
            time.sleep(0.1)
    return tasks


def run_smacv2(p, seed):
    tasks = []
    for env_name in ["protoss", "terran", "zerg"]:
        for mode in ["5_vs_5", "10_vs_10", "10_vs_11", "20_vs_20", "20_vs_23"]:
            tasks.append(p.submit(run_cmd, f"{env_name}_{mode}", "medium", seed, 50))
            time.sleep(0.1)
    return tasks


def run(seeds):
    with Pool(4) as p:
        tasks = []
        for seed in seeds:
            tasks += run_smacv2(p, seed)
            tasks += run_smacv1(p, seed)
            tasks += run_mamujoco(p, seed)
        for task in tasks:
            task.result()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default="0")
    args = parser.parse_args()
    
    run(args.seed.split(","))
    