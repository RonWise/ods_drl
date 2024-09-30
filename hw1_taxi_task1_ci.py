"""CI mode for RL agent training. HW1 Taxi, Task 1"""
# ruff: noqa: F821

import click  # noqa
import gym  # noqa
import numpy as np  # noqa

env = gym.make("Taxi-v3")

ACTION_N = 6
STATE_N = 500
LOG_PATH = "./output"


class CrossEntropyAgent:
    def __init__(self, state_n, action_n, q_param):
        self.model = np.ones((state_n, action_n)) / action_n
        self.state_n = state_n
        self.action_n = action_n
        self.q_param = q_param

    def get_action(self, state):
        return int(np.random.choice(np.arange(self.action_n), p=self.model[state]))

    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) == 0:
                new_model[state] = self.model[state]
            else:
                new_model[state] /= np.sum(new_model[state])
        self.model = new_model


def get_state(observation):
    return observation


def sample_trajectory(env, agent, trajectory_len=1000, visualization=False):
    trajectory = {"states": [], "actions": [], "rewards": []}

    observation = env.reset()

    for _ in range(trajectory_len):
        state = get_state(observation)
        trajectory["states"].append(state)

        action = agent.get_action(state)
        trajectory["actions"].append(action)

        observation, reward, done, _ = env.step(action)
        trajectory["rewards"].append(reward)

        if done:
            break

        if visualization:
            env.render()
            # time.sleep(0.1)

    return trajectory


def train(q_param, trajectory_len, trajectory_n, iteration_n, log=False):
    agent = CrossEntropyAgent(STATE_N, ACTION_N, q_param)

    mean_total_rewards = []
    for _ in range(iteration_n):
        # policy evaluation
        trajectories = [
            sample_trajectory(env, agent, trajectory_len=trajectory_len)
            for _ in range(trajectory_n)
        ]
        total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
        mean_total_rewards.append(np.mean(total_rewards))

        # policy improvement
        quantile = np.quantile(total_rewards, q=q_param)
        elite_trajectories = []
        for trajectory, total_reward in zip(trajectories, total_rewards):
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        agent.fit(elite_trajectories)
    if log:
        file_name = (
            LOG_PATH
            + "/"
            + str(int(100 * q_param))
            + "_"
            + str(trajectory_len)
            + "_"
            + str(trajectory_n)
            + "_"
            + str(iteration_n)
            + ".txt"
        )
        with open(file_name, "w") as outfile:
            outfile.write(",".join(map(str, np.around(mean_total_rewards, 2).tolist())))


@click.command()
@click.option("--q_param", default=0.9, help="quantile parameter")
@click.option("--trajectory_len", default=1000, help="trajectory length")
@click.option("--trajectory_n", default=1000, help="number of trajectories")
@click.option("--iteration_n", default=50, help="number of epochs")
@click.option("--log", default=True, help="flag to log results")
@click.option("--verbosity", default=False, help="flag to print worker info")
def run_train(q_param, trajectory_len, trajectory_n, iteration_n, log, verbosity):
    if verbosity:
        print(q_param, trajectory_len, trajectory_n, iteration_n, log)
    train(q_param, trajectory_len, trajectory_n, iteration_n, log)


if __name__ == "__main__":
    run_train()
