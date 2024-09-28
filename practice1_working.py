import time

import gym  # noqa
import gym_maze  # noqa

env = gym.make("maze-sample-5x5-v0")

state = env.reset()
print(state)
env.render()
time.sleep(3)
