from .envs.util import Color
from .envs.gomoku import GomokuEnv, GomokuState, Board
from gym.envs.registration import register

register(
    id='Gomoku19x19-v0',
    entry_point='gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': Color.black,
        'opponent': 'beginner',  # beginner opponent policy has defend and strike rules
        'board_size': 19,
    },
    nondeterministic=True,
)

register(
    id='Gomoku19x19-v1',
    entry_point='gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': None,
        'opponent': 'self',  # beginner opponent policy has defend and strike rules
        'board_size': 19,
    },
    nondeterministic=True,
)

register(
    id='Gomoku9x9-v0',
    entry_point='gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': Color.black,
        'opponent': 'beginner', # random policy is the simplest
        'board_size': 9,
    },
    nondeterministic=True,
)

