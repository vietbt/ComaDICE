import os
import torch
import random
import numpy as np
from runner import TRAINER_DISCRETE, TRAINER_CONTINUOUS


def run_smac(dataset, env_name, mode, algo, seed=0, alpha=10, n_epochs=55, f_divergence="soft_chisquare", lr=5e-4, device="cuda"):
    logdir = f"logs/{algo}/{env_name}/{mode}/seed{seed}/alpha{alpha}/{f_divergence}"
    os.makedirs(logdir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    st_dim = dataset.st_dim
    ob_dim = dataset.ob_dim
    ac_dim = dataset.ac_dim
    n_agents = dataset.n_agents
    trainer_cls = TRAINER_DISCRETE[algo]

    if algo == "madice":
        from agents.discrete import MADice
        agent = MADice(st_dim, ob_dim, ac_dim, n_agents).to(device)
        trainer = trainer_cls(agent, logdir, dataset, n_agents, alpha, lr, f_divergence, seed, device)
    else:
        trainer = trainer_cls(logdir, dataset, n_agents, lr, seed, device)
    trainer.train(n_epochs)


def run_mamujoco(dataset, env_name, mode, algo, seed=0, alpha=10, n_epochs=25, f_divergence="soft_chisquare", lr=5e-4, device="cuda"):
    logdir = f"logs/{algo}/{env_name}/{mode}/seed{seed}/alpha{alpha}/{f_divergence}"
    os.makedirs(logdir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    st_dim = dataset.st_dim
    ob_dim = dataset.ob_dim
    ac_dim = dataset.ac_dim
    n_agents = dataset.n_agents
    trainer_cls = TRAINER_CONTINUOUS[algo]

    if algo == "madice":
        from agents.continuous.madice import MADice
        agent = MADice(st_dim, ob_dim, ac_dim, n_agents).to(device)
        trainer = trainer_cls(agent, logdir, dataset, n_agents, alpha, lr, f_divergence, seed, device)
    else:
        trainer = trainer_cls(logdir, dataset, n_agents, lr, seed, device)
    trainer.train(n_epochs)


def main(env_name, mode, seed, n_epochs, lr, device):
    if any([name in env_name for name in ["protoss", "terran", "zerg", "5m_vs_6m", "2c_vs_64zg", "6h_vs_8z", "corridor"]]):
        from runner.data import SMACDataset
        dataset = SMACDataset.load(env_name, mode)
        is_discrete = True
    elif any([name in env_name for name in ["Hopper-v2", "Ant-v2", "HalfCheetah-v2"]]):
        from runner.data import MaMujocoDataset
        dataset = MaMujocoDataset.load(env_name, mode)
        is_discrete = False
    else:
        raise ValueError(f"Unknown env_name: {env_name}")

    algo = "madice"
    alpha = 10
    f_divergence = "soft_chisquare"
    
    if is_discrete:
        run_smac(dataset, env_name, mode, algo, seed, alpha, n_epochs, f_divergence, lr, device)
    else:
        run_mamujoco(dataset, env_name, mode, algo, seed, alpha, n_epochs, f_divergence, lr, device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="5m_vs_6m")
    parser.add_argument("--mode", type=str, default="medium")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args.env_name, args.mode, args.seed, args.n_epochs, args.lr, args.device)