import gym
from gym import wrappers

import sys
sys.path.append('..')
import DeepReward

MAX_EPISODES = 20
MAX_STEPS = 100
outdir = 'gridworld-results'

if __name__ == "__main__":
    env = gym.make('GridWorld-maczniak-v0')
    env = wrappers.Monitor(env, directory=outdir, force=True)

    for i_episode in range(MAX_EPISODES):
        observation = env.reset()
        env.render()
        for t in range(MAX_STEPS):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            env.render()
            if done:
                print("Episode finished after %d timesteps" % (t + 1))
                print()
                break

