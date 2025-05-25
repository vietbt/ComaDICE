import os
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# from concurrent.futures import ThreadPoolExecutor as Pool
from concurrent.futures import ProcessPoolExecutor as Pool
import pandas as pd
import numpy as np


def read_tensorboard(logdir, env_name, mode, seed, alpha, f_func, tags=["game/return", "game/winrate"]):
    logdir = f"{logdir}/{env_name}/{mode}/{seed}/{alpha}/{f_func}"
    files = os.listdir(logdir)
    if len(files) == 0:
        return
    path = f"{logdir}/{files[0]}"
    tb = EventAccumulator(path)
    tb.Reload()
    tags = [tag for tag in tags if tag in tb.Tags()["scalars"]]
    if len(tags) == 0:
        return
    data = defaultdict(list)
    for tag in tags:
        for item in tb.Scalars(tag):
            data[item.step].append(item.value)
    return data, env_name, mode, alpha, f_func


def main():
    # print("Hello from analyze.py")
    
    logdir = "logs/madice"
    # logdir = "logs_ori/madice"

    all_data = defaultdict(lambda: defaultdict(list))
    with Pool() as p:
        tasks = []
        for env_name in os.listdir(logdir):
            for mode in os.listdir(f"{logdir}/{env_name}"):
                for seed in os.listdir(f"{logdir}/{env_name}/{mode}"):
                    for alpha in os.listdir(f"{logdir}/{env_name}/{mode}/{seed}"):
                        for f_func in os.listdir(f"{logdir}/{env_name}/{mode}/{seed}/{alpha}"):
                            tasks.append((p.submit(read_tensorboard, logdir, env_name, mode, seed, alpha, f_func)))
        for task in tasks:
            results = task.result()
            if results is not None:
                data, env_name, mode, alpha, f_func = results
                for step, score in data.items():
                    if step > 100:
                        continue
                    all_data[(env_name, mode, alpha, f_func)][step].append(score)

    scores = {}
    for key in sorted(all_data.keys()):
        values = all_data[key]
        max_len = max(len(v) for v in values.values())
        values = {k: v for k, v in values.items() if len(v) == max_len}
        last_step = max(values.keys())
        values = values[last_step]
        try:
            last_value = np.array(values).mean(0)
            last_value = [round(v, 3) for v in last_value]
            scores[key] = (last_step, last_value, len(values))
        except:
            pass

    df = pd.DataFrame(scores, index=["step", "score", "n_seeds"]).T
    df["step"] = df["step"].astype(int)
    df["n_seeds"] = df["n_seeds"].astype(int)
    print(df.to_string())


if __name__ == "__main__":
    main()