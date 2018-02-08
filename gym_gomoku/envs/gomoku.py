from typing import List, Tuple

import numpy as np
import gym
from gym import spaces
from gym import error
from gym.utils import seeding
from six import StringIO
import sys, os
import six

from gym_gomoku.envs.util import gomoku_util, Color
from gym_gomoku.envs.util import make_random_policy
from gym_gomoku.envs.util import make_beginner_policy
from gym_gomoku.envs.util import make_medium_policy
from gym_gomoku.envs.util import make_expert_policy

# Rules from Wikipedia: Gomoku is an abstract strategy board game, Gobang or Five in a Row, it is traditionally played with Go pieces (black and white stones) on a go board with 19x19 or (15x15) 
# The winner is the first player to get an unbroken row of five stones horizontally, vertically, or diagonally. (so-calle five-in-a row)
# Black plays first if white did not win in the previous game, and players alternate in placing a stone of their color on an empty intersection.



class Board(object):
    '''
    Basic Implementation of a Go Board, natural action are int [0,board_size**2)
    '''
    def __init__(self, board_size: int):
        self.size = board_size
        self.board_state = [[Color.empty.value] * board_size for i in range(board_size)]  # initialize board states to empty
        self.move = 0                 # how many move has been made
        self.last_coord = (-1, -1)     # last action coord
        self.last_action = None       # last action made
        self.position_sets = {Color.empty: set(range(board_size**2)),
                              Color.black: set(),
                              Color.white: set()}

    @property
    def valid_actions(self):
        return self.position_sets[Color.empty]

    def coord_to_action(self, i, j):
        ''' convert coordinate i, j to action a in [0, board_size**2)
        '''
        a = i * self.size + j  # action index
        return a

    def action_to_coord(self, a):
        coord = (a // self.size, a % self.size)
        return coord

    def get_legal_move(self):
        ''' Get all the next legal move, namely empty space that you can place your 'color' stone
            Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
        '''
        legal_move = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    legal_move.append((i, j))
        return legal_move

    def get_legal_action(self):
        ''' Get all the next legal action, namely empty space that you can place your 'color' stone
            Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
        '''
        legal_action = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    legal_action.append(self.coord_to_action(i, j))
        return legal_action

    def copy(self, board_state, position_sets):
        '''update board_state of current board values from input 2D list
        '''
        input_size_x = len(board_state)
        input_size_y = len(board_state[0])
        assert input_size_x == input_size_y, 'input board_state two axises size mismatch'
        assert len(self.board_state) == input_size_x, 'input board_state size mismatch'
        for i in range(self.size):
            for j in range(self.size):
                self.board_state[i][j] = board_state[i][j]

        # copy the position_sets
        self.position_sets = dict([(k, set(v)) for k, v in position_sets.items()])

    def play(self, action, color):
        '''
            Args: input action, current player color
            Return: new copy of board object
        '''
        b = Board(self.size)
        b.copy(self.board_state, self.position_sets)  # create a board copy of current board_state
        b.move = self.move

        coord = self.action_to_coord(action)
        # check if it's legal move
        if b.board_state[coord[0]][coord[1]] != 0:  # the action coordinate is not empty
            raise error.InvalidAction("Action is illegal, position [%d, %d] on board is not empty" % ((coord[0]+1),(coord[1]+1)))
        # check if it's legal move
        if action not in b.position_sets[Color.empty]:
            raise error.InvalidAction("Action is illegal, position [%d, %d] on board is not empty" % ((coord[0] + 1), (coord[1] + 1)))

        # make move
        b.position_sets[Color.empty].remove(action)
        b.position_sets[color].add(action)

        b.board_state[coord[0]][coord[1]] = color.value
        b.move += 1  # move counter add 1
        b.last_coord = coord  # save last coordinate
        b.last_action = action
        return b

    def is_terminal(self):
        exist, color = gomoku_util.check_five_in_row(self.board_state)
        is_full = gomoku_util.check_board_full(self.board_state)
        if (is_full): # if the board if full of stones and no extra empty spaces, game is finished
            return True
        else:
            return exist

    def __repr__(self):
        ''' representation of the board class
            print out board_state
        '''
        out = ""
        size = len(self.board_state)

        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:size]

        label_move = "Move: " + str(self.move) + "\n"
        label_letters = "     " + " ".join(letters) + "\n"
        label_letters = "     " + " ".join(letters) + "\n"
        label_boundry = "   " + "+-" + "".join(["-"] * (2 * size)) + "+" + "\n"

        # construct the board output
        out += (label_move + label_letters + label_boundry)

        for i in range(size-1,-1,-1):
            line = ""
            line += (str("%2d" % (i+1)) + " |" + " ")
            for j in range(size):
                # check if it's the last move
                line += Color(self.board_state[i][j]).shape
                if (i, j) == self.last_coord:
                    line += ")"
                else:
                    line += " "
            line += ("|" + "\n")
            out += line
        out += (label_boundry + label_letters)
        return out

    def encode(self):
        ''' Return: np array
            np.array(board_size, board_size): state observation of the board
        '''
        img = np.array(self.board_state) # shape [board_size, board_size]
        return img


class GomokuState(object):
    '''
    Similar to Go game, Gomoku state consists of a current player and a board.
    Actions are exposed as integers in [0, num_actions), which is to place stone on empty intersection
    '''
    def __init__(self, board: Board, color: Color):
        '''
        Args:
            board: current board
            color: color of current player
        '''
        assert color in Color.players(), 'Invalid player color'
        self.board, self.color = board, color

    def act(self, action):
        '''
        Executes an action for the current player
        
        Returns:
            a new GomokuState with the new board and the player switched
        '''
        return GomokuState(self.board.play(action, self.color), self.color.other)

    @property
    def empty(self):
        return self.board.position_sets[Color.empty]

    @property
    def mine(self):
        return self.board.position_sets[self.color]

    @property
    def others(self):
        return self.board.position_sets[self.color.other]

    def coord_to_action(self, i, j=None) -> int:
        ''' convert coordinate i, j to action a in [0, board_size**2)
        '''
        if j is None:
            i, j = i
        a = i * self.board.size + j  # action index
        return a

    def action_to_coord(self, a: int) -> Tuple[int, int]:
        coord = (a // self.board.size, a % self.board.size)
        return coord

    def __repr__(self):
        '''stream of board shape output'''
        # To Do: Output shape * * * o o
        return 'To play: {}\n{}'.format(self.color.name, self.board.__repr__())


# Sampling without replacement Wrapper 
# sample() method will only sample from valid spaces
class GomokuActionSpace(spaces.Discrete):
    def __init__(self, n):
        self.n = n
        self.valid_spaces = list(range(n))  # type: List[int]
    
    def sample(self):
        '''Only sample from the remaining valid spaces
        '''
        if len(self.valid_spaces) == 0:
            print("Space is empty")
            return None
        np_random, _ = seeding.np_random()
        randint = np_random.randint(len(self.valid_spaces))
        return self.valid_spaces[randint]
    
    def remove(self, s):
        '''Remove space s from the valid spaces
        '''
        if s is None:
            return
        if s in self.valid_spaces:
            self.valid_spaces.remove(s)
        else:
            print("space %d is not in valid spaces" % s)

    # return all valid actions
    def actions(self):
        return self.valid_spaces

### Environment
class GomokuEnv(gym.Env):
    '''
    GomokuEnv environment. Play against a fixed opponent.
    '''
    metadata = {"render.modes": ["human", "ansi"]}
    reward_range = (-1., 1.)
    
    def __init__(self, player_color, opponent, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
                          if opponent is 'self' then this value is discarded.
            opponent: Name of the opponent policy, e.g. random, beginner, medium, expert
                      if 'self' then you play both players.
            board_size: board_size of the board to use
        """
        self.board_size = board_size
        self.player_color = player_color
        
        self.seed()
        
        # opponent
        self.opponent_policy = None
        self.opponent = opponent
        
        # Observation space on board
        shape = (self.board_size, self.board_size)  # board_size * board_size
        self.observation_space = spaces.Box(low=0, high=2, shape=shape, dtype=np.uint8)
        
        # One action for each board position
        self.action_space = GomokuActionSpace(self.board_size ** 2)
        
        # Keep track of the moves
        self.moves = []
        
        # Empty State
        self.state = None  # Type: Optional[GomokuState]
        
        # reset the board during initialization
        self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]
    
    def reset(self):
        self.state = GomokuState(Board(self.board_size), Color.black)  # Black Plays First
        self.reset_opponent(self.state.board)  # (re-initialize) the opponent,
        self.moves = []

        if self.opponent_policy:
            # Let the opponent play if it's not the agent's turn, there is no resign in Gomoku
            if self.state.color != self.player_color:
                self.state, _ = self.exec_opponent_play(self.state, None, None)
                opponent_action_coord = self.state.board.last_coord
                self.moves.append(opponent_action_coord)
        
            # We should be back to the agent color
            assert self.state.color == self.player_color
        
        # reset action_space
        self.action_space = GomokuActionSpace(self.board_size ** 2)
        
        self.done = self.state.board.is_terminal()
        return self.state.board.encode()
    
    def close(self):
        self.opponent_policy = None
        self.state = None
    
    def render(self, mode="human"):
        if mode == "human":
            print(self.state)
        elif mode == "ansi":
            outfile = StringIO()
            outfile.write(str(self.state) + '\n')
            return outfile
        else:
            super(GomokuEnv, self).render(mode=mode)
    
    def step(self, action):
        '''
        Args: 
            action: int
        Return: 
            observation: board encoding, 
            reward: reward of the game, 
            done: boolean, 
            info: state dict
        Raise:
            Illegal Move action, basically the position on board is not empty
        '''
        if self.opponent_policy:
            assert self.state.color == self.player_color # it's the player's turn
        
        # If already terminal, then don't do anything
        if self.done:
            return self.state.board.encode(), 0., True, {'state': self.state}
        
        # Player play
        prev_state = self.state
        self.state = self.state.act(action)
        self.moves.append(self.state.board.last_coord)
        self.action_space.remove(action)  # remove current action from action_space

        if self.opponent_policy:
            # Opponent play
            if not self.state.board.is_terminal():
                self.state, opponent_action = self.exec_opponent_play(self.state, prev_state, action)
                self.moves.append(self.state.board.last_coord)
                self.action_space.remove(opponent_action)   # remove opponent action from action_space
                # After opponent play, we should be back to the original color
                assert self.state.color == self.player_color
        
        # Reward: if nonterminal, there is no 5 in a row, then the reward is 0
        if not self.state.board.is_terminal():
            self.done = False
            return self.state.board.encode(), 0., False, {'state': self.state}
        
        # We're in a terminal state. Reward is 1 if won, -1 if lost
        assert self.state.board.is_terminal(), 'The game is terminal'
        self.done = True
        
        # Check Fianl wins
        exist, win_color = gomoku_util.check_five_in_row(self.state.board.board_state)  # 'empty', 'black', 'white'
        reward = 0.
        if win_color == "empty": # draw
            reward = 0.
        else:
            player_wins = (self.player_color == win_color) # check if player_color is the win_color
            reward = 1. if player_wins else -1.
        return self.state.board.encode(), reward, True, {'state': self.state}
    
    def exec_opponent_play(self, curr_state, prev_state, prev_action):
        '''There is no resign in gomoku'''
        assert curr_state.color != self.player_color
        opponent_action = self.opponent_policy(curr_state, prev_state, prev_action)
        return curr_state.act(opponent_action), opponent_action
    
    @property
    def _state(self):
        return self.state
    
    @property
    def _moves(self):
        return self.moves
    
    def reset_opponent(self, board):
        if self.opponent == 'random':
            self.opponent_policy = make_random_policy(self.np_random)
        elif self.opponent == 'beginner':
            self.opponent_policy = make_beginner_policy(self.np_random)
        elif self.opponent == 'medium':
            self.opponent_policy = make_medium_policy(self.np_random)
        elif self.opponent == 'expert':
            self.opponent_policy = make_expert_policy(self.np_random)
        elif self.opponent == 'self':
            self.opponent_policy = None
        else:
            raise error.Unregistered('Unrecognized opponent policy {}'.format(self.opponent))
