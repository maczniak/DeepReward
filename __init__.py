from gym.envs.registration import register
from GridWorldEnv import GridworldEnv

register(
    id='GridWorld-maczniak-v0',
    entry_point='DeepReward:GridworldEnv'
)

