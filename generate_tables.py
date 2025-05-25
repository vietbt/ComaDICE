import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from concurrent.futures import ProcessPoolExecutor as Pool
pd.options.display.max_rows = None

LOGDIR = "logs"
ALGOS = ["bc", "bcq", "cql", "icq", "omar", "omiga", "madice", "madice-twolayer"]
SEEDS = [0, 1, 2, 3, 4]
SMACV1_ENV_NAMES = ["2c_vs_64zg", "5m_vs_6m", "6h_vs_8z", "corridor"]
SMACV2_ENV_NAMES = ["protoss", "terran", "zerg"]
MAMUJOCO_ENV_NAMES = ["Hopper-v2", "Ant-v2", "HalfCheetah-v2"]
SMACV1_MODES = ["poor", "medium", "good"]
SMACV2_MODES = ["5_vs_5", "10_vs_10", "10_vs_11", "20_vs_20", "20_vs_23"]
MAMUJOCO_MODES = ["expert", "medium", "medium-replay", "medium-expert"]


def listdir(path):
    if not os.path.exists(path):
        return []
    return sorted(os.listdir(path))

def read_tensorboard(logdir):
    files = listdir(logdir)
    if len(files) == 0:
        return {}
    path = f"{logdir}/{files[0]}"
    tb = EventAccumulator(path)
    tb.Reload()
    data = {}
    for tag in ["return", "winrate"]:
        if f"game/{tag}" in tb.Tags()["scalars"]:
            for item in tb.Scalars(f"game/{tag}"):
                if item.step not in data:
                    data[item.step] = {}
                data[item.step][tag] = item.value
    return data

def create_tasks(algo, env_name, mode):
    for seed in listdir(f"{LOGDIR}/{algo}/{env_name}/{mode}"):
        for alpha in listdir(f"{LOGDIR}/{algo}/{env_name}/{mode}/{seed}"):
            for f_func in listdir(f"{LOGDIR}/{algo}/{env_name}/{mode}/{seed}/{alpha}"):
                logdir = f"{LOGDIR}/{algo}/{env_name}/{mode}/{seed}/{alpha}/{f_func}"
                yield logdir, alpha, f_func

def create_smacv1_tasks(p):
    tasks = []
    for algo in ALGOS:
        for env_name in SMACV1_ENV_NAMES:
            for mode in SMACV1_MODES:
                for logdir, alpha, f_func in create_tasks(algo, env_name, mode):
                    tasks.append((p.submit(read_tensorboard, logdir), env_name, mode, algo, alpha, f_func))
    return tasks

def create_smacv2_tasks(p):
    tasks = []
    for algo in ALGOS:
        for env_name in SMACV2_ENV_NAMES:
            for mode in SMACV2_MODES:
                for logdir, alpha, f_func in create_tasks(algo, f"{env_name}_{mode}", "medium"):
                    tasks.append((p.submit(read_tensorboard, logdir), env_name, mode, algo, alpha, f_func))
    return tasks

def create_mamujoco_tasks(p):
    tasks = []
    for algo in ALGOS:
        for env_name in MAMUJOCO_ENV_NAMES:
            for mode in MAMUJOCO_MODES:
                for logdir, alpha, f_func in create_tasks(algo, env_name, mode):
                    tasks.append((p.submit(read_tensorboard, logdir), env_name, mode, algo, alpha, f_func))
    return tasks

results = {}
with Pool() as p:
    tasks = []
    tasks += create_smacv1_tasks(p)
    tasks += create_smacv2_tasks(p)
    tasks += create_mamujoco_tasks(p)
    
    for task, env_name, mode, algo, alpha, f_func in tqdm(tasks):
        data = task.result()
        if len(data) == 0:
            continue
        if algo == "madice":
            algo = "ComaDICE (ours)"
        elif algo == "madice-twolayer":
            algo = "ComaDICE (2-layer)"
        else:
            algo = algo.upper()
        if (env_name, mode) not in results:
            results[(env_name, mode)] = {}
        if (algo, alpha, f_func) not in results[(env_name, mode)]:
            results[(env_name, mode)][(algo, alpha, f_func)] = []
        results[(env_name, mode)][(algo, alpha, f_func)].append(data)


main_results = {}
for (env_name, mode), data in results.items():
    for (algo, alpha, f_func), items in data.items():
        if alpha != "alpha10" or f_func != "soft_chisquare":
            continue
        if "2-layer" in algo:
            continue
        if (env_name, mode) not in main_results:
            main_results[(env_name, mode)] = {}
        main_results[(env_name, mode)][algo] = items


ablation_results = {}
for (env_name, mode), data in results.items():
    for (algo, alpha, f_func), items in data.items():
        if algo != "ComaDICE (ours)":
            continue
        if (env_name, mode) not in ablation_results:
            ablation_results[(env_name, mode)] = {}
        ablation_results[(env_name, mode)][(alpha, f_func)] = items


ablation_mixer_results = {}
for (env_name, mode), data in results.items():
    for (algo, alpha, f_func), items in data.items():
        if alpha != "alpha10" or f_func != "soft_chisquare":
            continue
        if "ComaDICE" not in algo:
            continue
        if (env_name, mode) not in ablation_mixer_results:
            ablation_mixer_results[(env_name, mode)] = {}
        ablation_mixer_results[(env_name, mode)][algo] = items


def analyze(items, tag="return"):
    if not isinstance(items, list):
        return np.nan
    items = [item[max(item.keys())] for item in items]
    items = [item[tag] for item in items if tag in item]
    if len(items) == 0:
        return np.nan
    mean = np.mean(items)
    std = np.std(items)
    if tag == "winrate":
        return f"{mean*100:.1f}±{std*100:.1f}"
    else:
        return f"{mean:.1f}±{std:.1f}"

def select_results(results, env_names=[], transpose=False, select_alpha=True):
    new_results = {}
    for (env_name, mode), data in results.items():
        if env_name not in env_names:
            continue
        if mode == "medium-replay":
            mode = "m-replay"
        elif mode == "medium-expert":
            mode = "m-expert"
        if transpose:
            for algo, items in data.items():
                if isinstance(algo, str):
                    algo = f"\\textbf{{{algo}}}"
                if (env_name, algo) not in new_results:
                    new_results[(env_name, algo)] = {}
                new_results[(env_name, algo)][mode] = items
        else:
            for algo, items in data.items():
                if algo in ["BC", "OMAR"]:
                    continue
                
                if isinstance(algo, tuple):
                    if select_alpha:
                        algo = algo[0]
                        algo = algo.replace("alpha", "\\alpha=")
                        algo = f"${algo}$"
                    else:
                        if algo[0] != "alpha10":
                            continue
                        algo = algo[1]
                else:
                    algo = f"\\textbf{{{algo}}}"
                if (env_name, mode) not in new_results:
                    new_results[(env_name, mode)] = {}
                new_results[(env_name, mode)][algo] = items
    return new_results
    

def print_latex(results, env_names, tag="return", caption="SMACv1", transpose=False, select_alpha=True):
    results = select_results(results, env_names, transpose, select_alpha)
    df = pd.DataFrame.from_dict(results).map(analyze, tag=tag).T
    latex = df.to_latex(position="htp", multicolumn=True, multirow=True)
    latex = latex.replace("\\begin{tabular}{l", "\small\n\centering\n\\begin{tabular}{c|")
    while "c|l" in latex:
        latex = latex.replace("c|l", "c|c|")
    
    for x in ["\cline{1-5}", "\cline{1-6}", "\cline{1-7}", "\cline{1-8}", "\cline{1-9}", "\cline{1-10}"]:
        latex = latex.replace(x, "\midrule")
    
    latex = latex.replace("\\midrule\n\\bottomrule", "\\bottomrule")
    latex = latex.replace("c|}", "c}")
    latex = latex.replace("_", "\_")
    latex = latex.replace("-v2}", "}")
    latex = latex.replace("{HalfCheetah}", "{\\shortstack{Half\\\\Cheetah}}")
    latex = latex.replace("multirow[t]", "multirow")
    latex = latex.replace(" &  & ", "\multicolumn{2}{c|}{\\textbf{Instances}} & ")
    latex = latex.replace(f"\\end{{table}}", f"\\caption{{{caption}}}\n\\end{{table}}")
    print(latex)


print("\subsection{Returns}")

print_latex(main_results, SMACV1_ENV_NAMES, "return", "SMACv1 - return")
print_latex(main_results, SMACV2_ENV_NAMES, "return", "SMACv2 - return")
print_latex(main_results, MAMUJOCO_ENV_NAMES, "return", "MaMujoco - return", transpose=True)

print("\subsection{Winrates}")

print_latex(main_results, SMACV1_ENV_NAMES, "winrate", "SMACv1 - winrate")
print_latex(main_results, SMACV2_ENV_NAMES, "winrate", "SMACv2 - winrate")

print("\subsection{Returns}")

print_latex(ablation_results, SMACV1_ENV_NAMES, "return", "SMACv1 - return - alpha", select_alpha=True)
print_latex(ablation_results, SMACV2_ENV_NAMES, "return", "SMACv2 - return - alpha", select_alpha=True)
print_latex(ablation_results, MAMUJOCO_ENV_NAMES, "return", "MaMujoco - return - alpha", select_alpha=True)


print("\subsection{Winrates}")
print_latex(ablation_results, SMACV1_ENV_NAMES, "winrate", "SMACv1 - winrate - alpha", select_alpha=True)
print_latex(ablation_results, SMACV2_ENV_NAMES, "winrate", "SMACv2 - winrate - alpha", select_alpha=True)


print("\subsection{Returns}")
print_latex(ablation_results, SMACV1_ENV_NAMES, "return", "SMACv1 - return - fdiv", select_alpha=False)
print_latex(ablation_results, SMACV2_ENV_NAMES, "return", "SMACv2 - return - fdiv", select_alpha=False)
print_latex(ablation_results, MAMUJOCO_ENV_NAMES, "return", "MaMujoco - return - fdiv", select_alpha=False)

print("\subsection{Winrates}")
print_latex(ablation_results, SMACV1_ENV_NAMES, "winrate", "SMACv1 - winrate - fdiv", select_alpha=False)
print_latex(ablation_results, SMACV2_ENV_NAMES, "winrate", "SMACv2 - winrate - fdiv", select_alpha=False)

print("\subsection{MaMujoco}")
print_latex(main_results, MAMUJOCO_ENV_NAMES, "return", "MaMujoco - return")

print("\subsection{Returns}")
print_latex(ablation_mixer_results, SMACV1_ENV_NAMES, "return", "SMACv1 - return - 2-layer")
print_latex(ablation_mixer_results, SMACV2_ENV_NAMES, "return", "SMACv2 - return - 2-layer")
print_latex(ablation_mixer_results, MAMUJOCO_ENV_NAMES, "return", "MaMujoco - return - 2-layer")

print("\subsection{Winrates}")
print_latex(ablation_mixer_results, SMACV1_ENV_NAMES, "winrate", "SMACv1 - winrate - 2-layer")
print_latex(ablation_mixer_results, SMACV2_ENV_NAMES, "winrate", "SMACv2 - winrate - 2-layer")