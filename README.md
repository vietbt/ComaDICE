# ComaDICE: Offline Cooperative Multi-Agent Reinforcement Learning with Stationary Distribution Shift Regularization
Offline reinforcement learning (RL) has garnered significant attention for its ability to learn effective policies from pre-collected datasets without the need for further environmental interactions. While promising results have been demonstrated in single-agent settings, offline multi-agent reinforcement learning (MARL) presents additional challenges due to the large joint state-action space and the complexity of multi-agent behaviors. A key issue in offline RL is the *distributional shift*, which arises when the target policy being optimized deviates from the behavior policy that generated the data. This problem is exacerbated in MARL due to the interdependence between agents' local policies and the expansive joint state-action space. Prior approaches have primarily addressed this challenge by incorporating regularization in the space of either Q-functions or policies. In this work, we introduce a regularizer in the space of stationary distributions to better handle distributional shift. Our algorithm, ComaDICE, offers a principled framework for offline cooperative MARL by incorporating stationary distribution regularization for the global learning policy, complemented by a carefully structured multi-agent value decomposition strategy to facilitate  multi-agent training. Through extensive experiments on the multi-agent *MuJoCo* and *StarCraft II* benchmarks, we demonstrate that ComaDICE achieves superior performance compared to state-of-the-art offline MARL methods across nearly all tasks.


# Installation
- SMACv1: Check this repo https://github.com/oxwhirl/smac/
- SMACv2: Check this repo https://github.com/oxwhirl/smacv2/
- MaMujoco: Check this repo https://github.com/oxwhirl/facmac/

# Offline Demonstrations
- SMACv1: https://cloud.tsinghua.edu.cn/d/f3c509d7a9d54ccd89c4/
- SMACv2: Contact us
- MaMujoco: https://cloud.tsinghua.edu.cn/d/dcf588d659214a28a777/

# Train Scripts
- Run `python -u main.py --env_name <env_name> --mode <mode> --seed <seed> --n_epochs <n_epochs>` for training the task *<env_name>* with scienarios *<mode>*
- For examples, for training SMACv2 tasks, run the following scripts:
    - `python -u main.py --env_name protoss_5_vs_5 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name protoss_10_vs_10 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name protoss_10_vs_11 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name protoss_20_vs_20 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name protoss_20_vs_23 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name terran_5_vs_5 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name terran_10_vs_10 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name terran_10_vs_11 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name terran_20_vs_20 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name terran_20_vs_23 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name zerg_5_vs_5 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name zerg_10_vs_10 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name zerg_10_vs_11 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name zerg_20_vs_20 --mode medium --seed 0 --n_epochs 50`
    - `python -u main.py --env_name zerg_20_vs_23 --mode medium --seed 0 --n_epochs 50`
- To train MaMujoco tasks:
    - `python -u main.py --env_name Hopper-v2 --mode expert --seed 0 --n_epochs 25`
    - `python -u main.py --env_name Hopper-v2 --mode medium --seed 0 --n_epochs 25`
    - `python -u main.py --env_name Hopper-v2 --mode medium-replay --seed 0 --n_epochs 25`
    - `python -u main.py --env_name Hopper-v2 --mode medium-expert --seed 0 --n_epochs 25`
    - `python -u main.py --env_name Ant-v2 --mode expert --seed 0 --n_epochs 25`
    - `python -u main.py --env_name Ant-v2 --mode medium --seed 0 --n_epochs 25`
    - `python -u main.py --env_name Ant-v2 --mode medium-replay --seed 0 --n_epochs 25`
    - `python -u main.py --env_name Ant-v2 --mode medium-expert --seed 0 --n_epochs 25`
    - `python -u main.py --env_name HalfCheetah-v2 --mode expert --seed 0 --n_epochs 25`
    - `python -u main.py --env_name HalfCheetah-v2 --mode medium --seed 0 --n_epochs 25`
    - `python -u main.py --env_name HalfCheetah-v2 --mode medium-replay --seed 0 --n_epochs 25`
    - `python -u main.py --env_name HalfCheetah-v2 --mode medium-expert --seed 0 --n_epochs 25`

- All experiments will be logged into `logs` folder during training.

# Evaluation and Visualization
- Check the notebook `analyze.ipynb` for evaluation and visualization
- We're also sharing *saved results* (compressed as a pickle file) for reproductivity

# Contact us
- Will be revealed later