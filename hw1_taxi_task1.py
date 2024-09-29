import time  # noqa

import gym  # noqa
import gym_maze  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

env = gym.make("Taxi-v3")

action_n = 6
state_n = 500


class RandomAgent:
    def __init__(self, action_n):
        self.action_n = action_n

    def get_action(self, state=None):
        return np.random.randint(self.action_n)


class CrossEntropyAgent:
    def __init__(self, state_n, action_n, q_param):
        self.model = np.ones((state_n, action_n)) / action_n
        self.state_n = state_n
        self.acton_n = action_n
        self.q_param = q_param

    def get_action(self, state):
        return int(np.random.choice(np.arange(self.acton_n), p=self.model[state]))

    def fit(self, trajectories):
        new_model = np.zeros((state_n, action_n))
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
            time.sleep(0.2)

    return trajectory


iteration_n = 1
q_param = 0.9
trajectory_n = 20

# agent = CrossEntropyAgent(state_n, action_n, q_param)

observation = env.reset()
taxi_row, taxi_col, passenger_location, destination = env.decode(observation)
print(f"{observation}: {taxi_row}, {taxi_col}, {passenger_location}, {destination}")


# agent = RandomAgent(action_n)
# sample_trajectory(env, agent, visualization=True)

# mean_total_rewards = []
# for iteration in range(iteration_n):
#     # policy evaluation
#     trajectories = [sample_trajectory(env, agent) for _ in range(trajectory_n)]
#     total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
#     mean_total_rewards.append(np.mean(total_rewards))
#     print(f"#{iteration}: {np.mean(total_rewards)}")
#     # policy improvement
#     quantile = np.quantile(total_rewards, q=q_param)
#     elite_trajectories = []
#     for trajectory, total_reward in zip(trajectories, total_rewards):
#         if total_reward > quantile:
#             elite_trajectories.append(trajectory)

#     agent.fit(elite_trajectories)

# plt.plot(mean_total_rewards)
# plt.show()
# sample_trajectory(env, agent, visualization=True)
