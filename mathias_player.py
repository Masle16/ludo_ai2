"""Mathias's Player
"""


import numpy as np
from pyludo.LudoGame import LudoGame, LudoState, LudoStateFull


class MathiasPlayer():
    """Genetic algorithm base class which is used as super for all developed genetic algorithm players
    """
    
    name = "MathiasPlayer"
    args = []
    inp_size = 4 * 59 + 1
    hidden_size = 100
    gene_count = inp_size * hidden_size + hidden_size

    def __init__(self):
        """Create a genetic algorihm player base
        
        Arguments:
            chromosome {np.array}
        """
        self.chromosome = np.load('mathias_player_chromosome.npy')
        w0_len = self.inp_size * self.hidden_size
        w1_len = self.hidden_size
        self.w0 = self.chromosome[:w0_len].reshape(self.inp_size, self.hidden_size)
        self.w1 = self.chromosome[w0_len:w0_len + w1_len].reshape(self.hidden_size)

    def play(self, state, dice_roll, next_states):
        """Play returns the token which is moved based on the eval actions function
        
        Arguments:
            state {LudoState}
            dice_roll {int}
            next_states {np.array(LudoState)}
        """
        full_state = LudoStateFull(state, dice_roll, next_states)
        action_values = self.eval_actions(full_state)
        actions_prioritized = np.argsort(-action_values)
        for token_id in actions_prioritized:
            if next_states[token_id] is not False:
                return token_id

    def eval_actions(self, full_state: LudoStateFull):
        """Evaluate actions
        
        Arguments:
            full_state {LudoStateFull} -- fulle state of the ludo game
        """
        action_scores = np.zeros(4)
        for action_id, state in enumerate(full_state.next_states):
            if state == False:
                action_scores[action_id] = -1e9
                continue
            flat_state_rep = np.zeros(self.inp_size)
            flat_state_rep[-1] = 1  # bias
            full_state_rep = flat_state_rep[:self.inp_size - 1].reshape((4, 59))
            for player_id in range(4):
                for token in state[player_id]:
                    full_state_rep[player_id][min(token + 1, 58)] += 1
            hidden = np.tanh((flat_state_rep @ self.w0) * np.sqrt(1 / self.inp_size))
            out = hidden @ self.w1
            action_scores[action_id] = out
        return action_scores


if __name__ == '__main__':
    import random
    import time
    from tqdm import tqdm
    from pyludo.StandardLudoPlayers import LudoPlayerRandom, SmartPlayer

    # players = [MathiasPlayer()] + [LudoPlayerRandom() for _ in range(3)]
    players = [MathiasPlayer()] + [SmartPlayer() for _ in range(3)]
    for i, player in enumerate(players):
        player.id = i

    win_rates = np.zeros(4)

    n = 1000

    start_time = time.time()
    bar = tqdm(range(n))
    for i in bar:
        random.shuffle(players)
        ludoGame = LudoGame(players)
        winner = ludoGame.play_full_game()
        win_rates[players[winner].id] += 1
        bar.set_description(f'Win rate {np.around(win_rates/np.sum(win_rates)*100, decimals=2)}')
    duration = time.time() - start_time

    print(f'win distribution {np.around(win_rates/np.sum(win_rates)*100, decimals=2)}')
    print(f'games per second {n / duration:.4f}')