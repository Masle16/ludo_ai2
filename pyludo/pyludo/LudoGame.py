import random
import logging
import numpy as np
from pyludo.utils import player_colors, star_jump, is_globe_pos

# cache the token to relative player part
tokens_relative_to_player = [[0 for _ in range(101)] for _ in range(4)]
for player_id in range(4):
    for token_pos in range(-1, 100):
        if token_pos == -1 or token_pos == 99:  # start and end pos are independent of player id
            new_pos = token_pos
        elif token_pos < 52:  # in common area
            new_pos = (token_pos - player_id * 13) % 52
        else:  # in end area, 52 <= x < 52 + 20
            new_pos = ((token_pos - 52 - player_id * 5) % 20) + 52
        tokens_relative_to_player[player_id][token_pos] = new_pos

# cache player id rotation for relative state computing
new_player_ids_table = []
for rel_player_id in range(4):
    new_player_ids_table.append([(x - rel_player_id) % 4 for x in range(4)])

next_player_cache = [1, 2, 3, 0]


class LudoState:
    def __init__(self, state=None, empty=False):
        if state is not None:
            self.state = state
        else:
            self.state = [[-1, -1, -1, -1] for _ in range(4)]

    def copy(self):
        return LudoState([tokens.copy() for tokens in self.state])

    def __getitem__(self, item):
        return self.state[item]

    def __setitem__(self, key, value):
        self.state[key] = value

    def __iter__(self):
        return self.state.__iter__()

    @staticmethod
    def get_tokens_relative_to_player(tokens, player_id):
        if player_id == 0:
            return tokens
        table = tokens_relative_to_player[player_id]
        return [table[token] for token in tokens]

    def get_state_relative_to_player(self, rel_player_id, keep_player_order=False):
        if rel_player_id == 0:
            return self
        rel = LudoState(empty=True)
        new_player_ids = [0, 1, 2, 3] if keep_player_order else new_player_ids_table[rel_player_id]
        for player_id, player_tokens in enumerate(self):
            new_player_id = new_player_ids[player_id]
            rel[new_player_id] = self.get_tokens_relative_to_player(player_tokens, rel_player_id)
        return rel

    def move_token(self, token_id, dice_roll, is_jump=False):
        """ move token for player 0 """
        cur_pos = self[0][token_id]
        if cur_pos == 99 or cur_pos == -1 and dice_roll != 6:
            return False  # can't move tokens in goal, and can only enter the board by rolling a 6

        new_state = self if is_jump else self.copy()
        player = new_state[0]
        opponents = new_state[1:]

        # start move
        if cur_pos == -1:
            player[token_id] = 1
            for opponent in opponents:
                for i, token in enumerate(opponent):
                    if token == 1:
                        opponent[i] = -1
            return new_state

        target_pos = cur_pos + dice_roll

        # common area move
        if target_pos < 52:
            occupants = []
            for opponent in opponents:
                for i, token in enumerate(opponent):
                    if token == target_pos:
                        occupants.append((opponent, i))
            occupant_count = len(occupants)
            if occupant_count > 1 or (occupant_count == 1 and is_globe_pos(target_pos)):
                player[token_id] = -1  # sends self home
                return new_state
            if occupant_count == 1:
                occupants[0][0][occupants[0][1]] = -1
            player[token_id] = target_pos
            star_jump_length = 0 if is_jump else star_jump(target_pos)
            if star_jump_length:
                if target_pos == 51:  # landed on the last star
                    player[token_id] = 99  # send directly to goal
                else:
                    new_state = new_state.move_token(token_id, star_jump_length, is_jump=True)
            return new_state

        # end zone move
        if target_pos == 57:  # token reached goal
            player[token_id] = 99
        elif target_pos < 57:  # no goal bounce
            player[token_id] = target_pos
        else:  # bounce back from goal pos
            player[token_id] = 57 - (target_pos - 57)
        return new_state

    def is_winner(self, player_id):
        for token in self.state[player_id]:
            if token != 99:
                return False
        return True

    def get_winner(self):
        for player_id in range(4):
            if self.is_winner(player_id):
                return player_id
        return -1


class LudoStateFull:
    def __init__(self, state, roll, next_states):
        self.state = state
        self.roll = roll
        self.next_states = next_states


class LudoGame:
    def __init__(self, players, state=None, info=False):
        assert len(players) == 4, "there must be four players in the game"
        self.players = players
        self.currentPlayerId = -1
        self.state = LudoState() if state is None else state

        self.info = info
        if info:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def step(self):
        state = self.state
        self.currentPlayerId = (self.currentPlayerId + 1) % 4
        player = self.players[self.currentPlayerId]

        dice_roll = random.randint(1, 6)
        if self.info:
            logging.info("Dice roll = {}, {}/{}".format(dice_roll, player_colors[self.currentPlayerId], player.name))

        relative_state = state.get_state_relative_to_player(self.currentPlayerId)
        rel_next_states = np.array(
            [relative_state.move_token(token_id, dice_roll) for token_id in range(4)]
        )
        if np.any(np.array(rel_next_states) != False):
            token_id = player.play(relative_state, dice_roll, rel_next_states)
            if isinstance(token_id, np.ndarray):
                token_id = token_id[0]
            if rel_next_states[token_id] is False:
                logging.warning("Player chose invalid move. Choosing first valid move.")
                token_id = np.argwhere(rel_next_states != False)[0][0]
            self.state = rel_next_states[token_id].get_state_relative_to_player((-self.currentPlayerId) % 4)

    def step_eff(self):
        relative_state = self.state
        self.currentPlayerId = next_player_cache[self.currentPlayerId]
        player = self.players[self.currentPlayerId]

        dice_roll = random.randint(1, 6)

        rel_next_states = [relative_state.move_token(token_id, dice_roll) for token_id in range(4)]
        is_valid_move = False
        for next_state in rel_next_states:
            if next_state is not False:
                is_valid_move = True
                break
        if is_valid_move:
            token_id = player.play(relative_state, dice_roll, rel_next_states)
            if rel_next_states[token_id] is False:
                logging.warning("Player chose invalid move. Choosing first valid move.")
                token_id = np.argwhere(np.array(rel_next_states) != False)[0][0]
            self.state = rel_next_states[token_id].get_state_relative_to_player(1)  # set relative to next playe
        else:
            self.state = self.state.get_state_relative_to_player(1)

    def play_full_game(self):
        if self.currentPlayerId == -1:
            self.step_eff()
        while not self.state.is_winner(3):
            self.step_eff()
        return self.currentPlayerId
