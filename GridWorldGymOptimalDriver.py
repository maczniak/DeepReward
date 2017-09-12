from __future__ import print_function
import gym
from gym import wrappers
import numpy as np

import sys
sys.path.append('..')
import DeepReward

MAX_EPISODES = 20
MAX_STEPS = 100
outdir = 'gridworld-results'

if __name__ == "__main__":
    env = gym.make('GridWorld-maczniak-v0')
    env = wrappers.Monitor(env, directory=outdir, force=True)

    env_ = env.env # hack - Monitor does not support DiscreteEnv yet
    nS = env.observation_space.n
    nA = env.action_space.n
    P = env_.P
    np_random = env_.np_random

    discount = 1.0
    nextState = np.empty((nS, nA), dtype=np.int)
    actionReward = np.empty((nS, nA))
    world = np.zeros((nS))
    for state in range(nS):
        for action in range(nA):
            # P[state][action] has only one (probability, nextstate, reward, done) entry
            _, nextState[state, action], actionReward[state, action], _ = \
                                                             P[state][action][0]

    # Here we can do value iteration without agent's experience because we know
    #  the model. In real world problems, it may be infeasible, and we need
    #  model-free methods.

    # value iteration
    while True:
        worldprime_withActions = actionReward + discount * world[nextState]
        worldprime = np.max(worldprime_withActions, axis = 1)
        if np.allclose(world, worldprime, 1e-4):
            world = worldprime
            print('optimal value function table:', world)
            print()
            break
        world = worldprime

    for i_episode in range(MAX_EPISODES):
        observation = env.reset() # observation - current state
        env.render()
        for t in range(MAX_STEPS):
            rewards = np.empty((nA))
            for action in range(nA): # instead of env.action_space
                _, nextstate, _, _ = P[observation][action][0]
                rewards[action] = world[nextstate]
            # follow the action that gives the most value
            action = np_random.choice(np.flatnonzero(rewards == rewards.max()))
            # do not use: action = np.argmax(rewards)
            # because np.argmax() always return the first index among indices
            #  that have the maximum value
            # and np_random is the seedable random generator of DiscreteEnv
            observation, reward, done, info = env.step(action)
            env.render()
            if done:
                print("Episode finished after %d timesteps" % (t + 1))
                print()
                break

