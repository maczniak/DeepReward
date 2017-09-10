# GridWorld environment (tweaked)
# original: https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py

import numpy as np
import sys
from gym.envs.toy_text import discrete
from gym.utils import colorize
import six

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
actions = [UP, RIGHT, DOWN, LEFT]

actionWords = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT"
}

class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.

    For example, a 4x4 grid looks as follows:

    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T

    x is your position and T are the two terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4,4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        MAX_Y = shape[0]
        MAX_X = shape[1]

        nS = MAX_Y * MAX_X # number of states
        nA = len(actions) # number of actions

        # transition, P[s][a] == [(probability, nextstate, reward, done), ...]
        P = {} # or P = np.empty(nS, dtype='O')

        state = lambda y, x: y * MAX_X + x
        is_done = lambda s: s == 0 or s == (nS - 1) # top left or bottom right

        for y in range(MAX_Y):
            for x in range(MAX_X):
                nextstate_up    = state(max(y - 1, 0),         x)
                nextstate_down  = state(min(y + 1, MAX_Y - 1), x)
                nextstate_right = state(y,                min(x + 1, MAX_X - 1))
                nextstate_left  = state(y,                max(x - 1, 0))

                P[state(y, x)] = {}
                P[state(y, x)][UP] = \
                    [(1.0, nextstate_up, -1.0, is_done(nextstate_up))]
                P[state(y, x)][DOWN] = \
                    [(1.0, nextstate_down, -1.0, is_done(nextstate_down))]
                P[state(y, x)][RIGHT] = \
                    [(1.0, nextstate_right, -1.0, is_done(nextstate_right))]
                P[state(y, x)][LEFT] = \
                    [(1.0, nextstate_left, -1.0, is_done(nextstate_left))]

        # We're stuck in a terminal state
        topleft_state = state(0, 0)
        P[topleft_state][UP]    = [(1.0, topleft_state, 0.0, True)]
        P[topleft_state][RIGHT] = [(1.0, topleft_state, 0.0, True)]
        P[topleft_state][DOWN]  = [(1.0, topleft_state, 0.0, True)]
        P[topleft_state][LEFT]  = [(1.0, topleft_state, 0.0, True)]
        bottomright_state = state(MAX_Y - 1, MAX_X - 1)
        P[bottomright_state][UP]    = [(1.0, bottomright_state, 0.0, True)]
        P[bottomright_state][RIGHT] = [(1.0, bottomright_state, 0.0, True)]
        P[bottomright_state][DOWN]  = [(1.0, bottomright_state, 0.0, True)]
        P[bottomright_state][LEFT]  = [(1.0, bottomright_state, 0.0, True)]

        # Initial state distribution is uniform
        isd = np.ones(nS) / (nS - 2)
        # prevent starting from the terminal states
        isd[state(0, 0)] = isd[state(MAX_Y - 1, MAX_X - 1)] = 0

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = six.StringIO() if mode == 'ansi' else sys.stdout

        shape = self.shape

        MAX_Y = shape[0]
        MAX_X = shape[1]

        state = lambda y, x: y * MAX_X + x

        if self.lastaction is None:
            outfile.write("started\n")
        else:
            outfile.write("did action %s\n" % actionWords[self.lastaction])

        for y in range(MAX_Y):
            for x in range(MAX_X):
                if state(y, x) == self.s: # self.s is provided by DiscreteEnv
                    output = colorize(" x ", "red", bold=True)
                elif state(y, x) == state(0, 0) or \
                        state(y, x) == state(MAX_Y - 1, MAX_X - 1):
                    output = colorize(" T ", "green")
                else:
                    output = " o "
                outfile.write(output)
            outfile.write("\n")
        outfile.write("\n")

        if mode == 'ansi':
            return outfile.getvalue()

