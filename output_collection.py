"""HW1 Taxi, Task 1. Collect results and store in pandas dataframe"""

# ruff: noqa: F821
import os  # noqa

import numpy as np  # noqa
import pandas as pd  # noqa

LOG_PATH = "./output"

results_path = os.getcwd() + "/" + LOG_PATH

data = []

for filename in os.listdir(results_path):
    print(filename)
    q, traj_len, traj_n, iter_n = map(int, filename.split(".")[0].split("_"))
    # print(q, traj_len, traj_n, iter_n)
    q_data = [q] * iter_n
    traj_len_data = [traj_len] * iter_n
    traj_n_data = [traj_n] * iter_n
    iter_n_data = [iter_n] * iter_n
    epoch_data = list(range(iter_n))
    with open(results_path + "/" + filename, "r") as file_input:
        line = file_input.read()
        result_data = list(map(float, line.split(",")))

    data.append(
        np.array(
            list(
                zip(
                    epoch_data,
                    q_data,
                    traj_len_data,
                    traj_n_data,
                    iter_n_data,
                    result_data,
                )
            )
        )
    )
    # break

data = np.concatenate(data, axis=0)

df = pd.DataFrame(
    data=data,
    columns=[
        "epoch",
        "q_param",
        "trajectory_len",
        "trajectory_n",
        "iteration_n",
        "mean_total_rewards",
    ],
)
df.set_index("epoch", inplace=True)
print(df.shape)

# store = pd.HDFStore(os.getcwd() + "/" + "hw1_taxi_task1_results.h5")
# store["data"] = df
df.to_pickle(os.getcwd() + "/" + "hw1_taxi_task1_results.pickle")
# df = pd.read_pickle(file_name)
