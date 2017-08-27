# GridWorld (tweaked)
# original: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter03/GridWorld.py

from __future__ import print_function
import numpy as np

WORLD_SIZE = 5
A_POS = (0, 1)
A_PRIME_POS = (4, 1)
B_POS = (0, 3)
B_PRIME_POS = (2, 3)
discount = 0.9

# left, up, right, down
LEFT  = 0
UP    = 1
RIGHT = 2
DOWN  = 3
actions = [LEFT, UP, RIGHT, DOWN]

actionProb   = np.full((WORLD_SIZE, WORLD_SIZE, len(actions)), 0.25)
nextState    = np.empty((WORLD_SIZE, WORLD_SIZE, len(actions)),
                        dtype=[('x', '<i8'), ('y', '<i8')])
actionReward = np.zeros((WORLD_SIZE, WORLD_SIZE, len(actions)))

for i in range(WORLD_SIZE):
    for j in range(WORLD_SIZE):
        next = nextState[i][j]
        reward = actionReward[i][j]

        next[UP]    = (max(i - 1, 0),              j)
        next[DOWN]  = (min(i + 1, WORLD_SIZE - 1), j)
        next[LEFT]  = (i,                          max(j - 1, 0))
        next[RIGHT] = (i,                          min(j + 1, WORLD_SIZE - 1))

        if i == 0:
            reward[UP] = -1.0

        if i == WORLD_SIZE - 1:
            reward[DOWN] = -1.0

        if j == 0:
            reward[LEFT] = -1.0

        if j == WORLD_SIZE - 1:
            reward[RIGHT] = -1.0

nextState[A_POS][LEFT] = nextState[A_POS][RIGHT] = \
 nextState[A_POS][DOWN] = nextState[A_POS][UP] = A_PRIME_POS
actionReward[A_POS][LEFT] = actionReward[A_POS][RIGHT] = \
 actionReward[A_POS][DOWN] = actionReward[A_POS][UP] = 10.0

nextState[B_POS][LEFT] = nextState[B_POS][RIGHT] = \
 nextState[B_POS][DOWN] = nextState[B_POS][UP] = B_PRIME_POS
actionReward[B_POS][LEFT] = actionReward[B_POS][RIGHT] = \
 actionReward[B_POS][DOWN] = actionReward[B_POS][UP] = 5.0

nextState_x = nextState['x']
nextState_y = nextState['y']

# for figure 3.5
world = np.zeros((WORLD_SIZE, WORLD_SIZE))

while True:
    # keep iteration until convergence
    # bellman equation
    newWorld_withActions = actionProb * \
               (actionReward + discount * world[nextState['x'], nextState['y']])
          # or (actionReward + discount * world[nextState_x,    nextState_y])
    newWorld = np.sum(newWorld_withActions, axis=2)
    if np.sum(np.abs(world - newWorld)) < 1e-4:
        print('Random Policy')
        print(newWorld)
        break
    world = newWorld

# for figure 3.8
world = np.zeros((WORLD_SIZE, WORLD_SIZE))

while True:
    # keep iteration until convergence
    # value iteration
    newWorld_withActions = actionReward + \
                                discount * world[nextState['x'], nextState['y']]
                           # or discount * world[nextState_x,    nextState_y]
    newWorld = np.max(newWorld_withActions, axis = 2)
    if np.sum(np.abs(world - newWorld)) < 1e-4:
        print('Optimal Policy')
        print(newWorld)
        break
    world = newWorld

