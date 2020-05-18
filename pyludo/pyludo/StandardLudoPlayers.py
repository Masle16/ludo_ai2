import random
import numpy as np
from .utils import token_vulnerability

"""
def play(self, state, dice_roll, next_states):
    :param state:
        current state relative to this player
    :param dice_roll:
        [1, 6]
    :param next_states:
        np array of length 4 with each entry being the next state moving the corresponding token.
        False indicates an invalid move. 'play' won't be called, if there are no valid moves.
    :return:
        index of the token that is wished to be moved. If it is invalid, the first valid token will be chosen.
"""


class LudoPlayerRandom:
    """ takes a random valid action """
    name = 'random'

    @staticmethod
    def play(state, dice_roll, next_states):
        actions = [0, 1, 2, 3]
        random.shuffle(actions)
        for action in actions:
            if next_states[action] is not False:
                return action


class LudoPlayerFast:
    """ moves the furthest token that can be moved """
    name = 'fast'

    @staticmethod
    def play(state, _, next_states):
        for token_id in np.argsort(state[0]):
            if next_states[token_id] is not False:
                return token_id


class LudoPlayerAggressive:
    """ tries to send the opponent home, else random valid move """
    name = 'aggressive'

    @staticmethod
    def play(state, dice_roll, next_states):
        for token_id, next_state in enumerate(next_states):
            if next_state is False:
                continue
            if np.sum(next_state[1:] == -1) > np.sum(state[1:] == -1):
                return token_id
        return LudoPlayerRandom.play(None, None, next_states)


class LudoPlayerDefensive:
    """ moves the token that can be hit by most opponents """
    name = 'defensive'

    @staticmethod
    def play(state, dice_roll, next_states):
        hit_rates = np.empty(4)
        hit_rates.fill(-1)
        for token_id, next_state in enumerate(next_states):
            if next_state is False:
                continue
            hit_rates[token_id] = token_vulnerability(state, token_id)
        return random.choice(np.argwhere(hit_rates == np.max(hit_rates)))

class SmartPlayer:
    """Smart player
    """

    name = "smart"

    @staticmethod
    def play(state, dice_roll, next_states):
        """[summary]
        
        Arguments:
            state {[type]} -- [description]
            dice_roll {[type]} -- [description]
            next_state {[type]} -- [description]
        """
        if dice_roll == 6:
            home_players = [i for i, token in enumerate(state[0]) if token == -1]
            if home_players:
                return random.choice(home_players)
        max_val = -1e9
        max_i = 0
        for i, state in enumerate(next_states):
            if state != False:
                val = sum(state[0]) - sum([sum(tokens) for tokens in state[1:]])
                if val > max_val:
                    max_val = val
                    max_i = i
        return max_i
