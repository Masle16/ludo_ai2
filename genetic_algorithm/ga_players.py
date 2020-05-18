"""Genetic algorithm players
"""


import time
import numpy as np
from pyludo.LudoGame import LudoGame, LudoState, LudoStateFull


class GAPlayerBase:
    """Genetic algorithm base class which is used as super for all developed genetic algorithm players
    """
    
    name = ""
    gene_count = None

    def __init__(self, chromosome):
        """Create a genetic algorihm player base
        
        Arguments:
            chromosome {np.array}
        """
        self.chromosome = chromosome
        self.pred_time = []

    def play(self, state, dice_roll, next_states):
        """Play returns the token which is moved based on the eval actions function
        
        Arguments:
            state {LudoState}
            dice_roll {int}
            next_states {np.array(LudoState)}
        """
        t1 = time.time()
        full_state = LudoStateFull(state, dice_roll, next_states)
        action_values = self.eval_actions(full_state)
        actions_prioritized = np.argsort(-action_values)
        for token_id in actions_prioritized:
            if next_states[token_id] is not False:
                t2 = time.time()
                self.pred_time.append((t2-t1))
                return token_id

    def eval_actions(self, full_state: LudoStateFull):
        """Is specific for each developed genetic algorithm player
        """
        pass

    @staticmethod
    def normalize(chromosome):
        """Is specific for each developed genetic algorithm player
        """
        return chromosome

    def pop_init(self, sigma_count, pop_size):
        """Initialize population
        
        Arguments:
            sigma_count {int} -- [description]
            pop_size {int} -- population size
        """
        genes = np.random.randn(pop_size * self.gene_count).reshape((pop_size, -1))
        sigma = np.exp(np.random.randn(pop_size * sigma_count) - 2).reshape((pop_size, -1))
        return np.concatenate((genes, sigma), axis=1)

    def star_jump(self, pos):
        """Check if the position is a star
        
        Arguments:
            pos {int} -- the position
        
        Returns:
            int -- 0 if it is not a star, otherwise > 0
        """
        if pos == -1 or pos > 51:
            return 0
        if pos % 13 == 6:
            return 6
        if pos % 13 == 12:
            return 7
        return 0

    def is_globe_pos(self, pos):
        """Checks if the position is a globe
        
        Arguments:
            pos {int} -- the position
        
        Returns:
            bool -- True if the position is a globe
        """
        if pos == -1 or pos > 51:
            return False
        if pos % 13 == 1:
            return True
        if pos % 13 == 9:
            return True
        return False
    
    @staticmethod
    def will_send_self_home(state, next_state):
        """Checks if the next state will send the ludo player home
        
        Arguments:
            state {LudoState} -- the current state of the game
            next_state {LudoState} -- the next state of the game
        
        Returns:
            bool -- true if more token are home than before
        """
        return np.sum(state[0] == -1) < np.sum(next_state[0] == -1)

    @staticmethod
    def will_send_opponent_home(state, next_state):
        """Checks if the next state will send other players home
        
        Arguments:
            state {LudoState} -- the current state of the game
            next_state {LudoState} -- the next state of the game
        
        Returns:
            bool -- true if more tokens are home than before
        """
        return np.sum(state[1:] == -1) < np.sum(next_state[1:] == -1)

    def token_vulnerability(self, state, token_id):
        """Returns an approximation of the amount, n, of opponent dice rolls that can send the token home
        
        Arguments:
            state {LudoState} -- the current state of the ludo game
            token_id {int} -- the current token
        
        Returns:
            int -- a approximation of the tokens position vulnerability
        """
        player = state[0]
        token = player[token_id]

        if token == -1 or token == 1 or token > 51:  # in home, start or end positions
            return 0
        if token % 13 == 1 and np.sum(state[token // 13] == -1) == 0:  # on globe outside empty home
            return 0
        if token % 13 != 1 and np.sum(player == token) > 1 or token % 13 == 9:  # blockade or globe
            return 0

        n = 0

        if token % 13 == 1:  # on opponent start pos
            n += 1

        star = self.star_jump(token)
        if star > 0:
            star = 6 if star == 7 else 7

        for opponent_id in range(1, 4):
            opponent = state[opponent_id]
            for opp_token in set(opponent):
                if opp_token == -1 or opp_token > 51:
                    continue
                req_dice_roll = (token - opp_token) % 52
                rel_opp_token = (opp_token - opponent_id * 13) % 52
                would_enter_end_zone = rel_opp_token + req_dice_roll > 51
                if not would_enter_end_zone and 1 <= req_dice_roll <= 6:
                    n += 1
                if star > 0:
                    req_dice_roll = (token - opp_token - star) % 52
                    would_enter_end_zone = rel_opp_token + req_dice_roll + star > 51
                    if not would_enter_end_zone and 1 <= req_dice_roll <= 6:
                        n += 1

        return n
    
    @staticmethod
    def count_home_tokens(opponents):
        """Count home tokens
        
        Arguments:
            opponents {LudoPlayer}
        
        Returns:
            int -- number of tokens which are home
        """
        home = 0
        for opp in opponents:
            for token in opp:
                if token == -1:
                    home += 1
        return home

    def is_free_globe(self, pos, opps):
        """Checks if position is a globe and no opponent is on it
        
        Arguments:
            pos {int} -- player position
            opps {np.array} -- position of opponents
        
        Returns:
            bool -- true if globe position is free
        """
        if not self.is_globe_pos(pos):
            return False

        for opp in opps:
            for opp_pos in opp:
                if pos == opp_pos:
                    return False
        
        return True


class GAPlayerSimple(GAPlayerBase):
    """Simple genetic algorithm player with 4 genes:
        1. Whether a player token was moved out from home and onto the board
        2. Whether a player toekn entered the safe zone
        3. Whether a player token entered goal
        4. Whether a opponent token was sent home
    
    Arguments:
        ga_player_base {ga_player_base} -- genetic algorithm player base
    """

    name = "Simple"
    args = []
    gene_count = 4

    def __init__(self, chromosome):
        """Creates a ga_player_simple obhect
        
        Arguments:
            chromosome {np.array}
        """
        super(GAPlayerSimple, self).__init__(chromosome)

    def eval_actions(self, full_state: LudoStateFull):
        """Evaluate actions
        
        Arguments:
            full_state {LudoStateFull}
        """
        action_scores = np.empty(4)
        state = full_state.state

        for token_id in range(4):
            next_state = full_state.next_states[token_id]
            if next_state == False:
                action_scores[token_id] = 0
                continue
            
            cur_token_pos = state[0][token_id]
            next_token_pos = next_state[0][token_id]
            cur_opponents = state[1:]
            next_opponents = next_state[1:]

            moved_out = (cur_token_pos == -1 and next_token_pos != -1)
            enter_goal = (cur_token_pos != 99 and next_token_pos == 99)
            enter_safe_zone = (cur_token_pos <= 51 and next_token_pos > 51)
            opps_hit_home = self.count_home_tokens(next_opponents) - self.count_home_tokens(cur_opponents)

            reduced_state = [
                moved_out,
                enter_goal,
                enter_safe_zone,
                opps_hit_home
            ]

            action_scores[token_id] = sum([gene * val for gene, val in zip(self.chromosome, reduced_state)])

        return action_scores
    
    @staticmethod
    def normalize(chromosome):
        """Normalize chromosome
        
        Arguments:
            chromosome {np.array}
        """
        gene_count = GAPlayerSimple.gene_count
        chromosome[:gene_count] /= np.abs(chromosome[:gene_count]).sum() * (1/gene_count)
        return chromosome


class GAPlayerFull(GAPlayerBase):
    """Artificial Neural Network genetic algorithm player
    
    Arguments:
        GAPlayerBase {GAPlayerBase} -- genetic algorithm player base
    """

    name = "Full"
    args = []
    inp_size = 4 * 59 + 1
    hidden_size = 100
    gene_count = (4 * 59 + 1) * 100 + 100

    def __init__(self, chromosome):
        """Creates a GAPlayerFull object
        
        Arguments:
            chromosome {np.array}
        """
        super(GAPlayerFull, self).__init__(chromosome)
        w0_len = self.inp_size * self.hidden_size
        w1_len = self.hidden_size
        self.w0 = chromosome[:w0_len].reshape(self.inp_size, self.hidden_size)
        self.w1 = chromosome[w0_len:w0_len + w1_len].reshape(self.hidden_size)

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
            out = np.tanh(hidden @ self.w1)
            action_scores[action_id] = out
        return action_scores


class GAPlayerAdvanced(GAPlayerBase):
    """Small ANN player

    Arguments:
        GAPlayerBase {object} -- Base GA player
    """

    name = 'Advanced'
    args = []
    inp_size = 4 * 59
    hidden_size = 4
    gene_count = inp_size * hidden_size + hidden_size

    def __init__(self, chromosome):
        """Creates a GAPlayerFull object
        
        Arguments:
            chromosome {np.array}
        """
        super(GAPlayerAdvanced, self).__init__(chromosome)
        w0_len = self.inp_size * self.hidden_size
        w1_len = self.hidden_size
        self.w0 = chromosome[:w0_len].reshape(self.inp_size, self.hidden_size)
        self.w1 = chromosome[w0_len:w0_len + w1_len].reshape(self.hidden_size)

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
            full_state_rep = flat_state_rep.reshape((4, 59))
            for player_id in range(4):
                for token in state[player_id]:
                    full_state_rep[player_id][min(token + 1, 58)] += 1
            hidden = np.tanh((flat_state_rep @ self.w0))
            out = np.tanh((hidden @ self.w1))
            action_scores[action_id] = out
        return action_scores


def get_ga_player(name):
    """Get genetic algorithm player
    
    Arguments:
        name {string} -- name of player
    """
    players = [GAPlayerAdvanced, GAPlayerSimple, GAPlayerFull]
    player_map = {}
    for player in players:
        player_map[player.name] = player
    return player_map[name]


if __name__ == '__main__':
    import random
    from tqdm import tqdm
    from pyludo.StandardLudoPlayers import LudoPlayerRandom

    game_count = 2500

    sim_chromo = np.load('genetic_algorithm/populations/Simple/Simple_Cellular-20-10_Blend-0.5_NStep/1000.pop.winner.npy')
    adv_chromo = np.load('genetic_algorithm/populations/Advanced/Advanced_Normal-100-10_Whole_Normal-0.1/1000.pop.winner.npy')
    full_chromo = np.load('genetic_algorithm/populations/Full/Full_Cellular-100-10_Whole_Normal-0.1/1000.pop.winner.npy')

    sim_player = GAPlayerSimple(sim_chromo)
    adv_player = GAPlayerAdvanced(adv_chromo)
    full_player = GAPlayerFull(full_chromo)
    rand_player = LudoPlayerRandom()

    players = [sim_player, adv_player, full_player, rand_player]

    f = open('genetic_algorithm/agent_evaluations/prediction_times.txt', 'w')
    
    tournament_player_ids = {}
    for i, player in enumerate(players):
        tournament_player_ids[player] = i
        print(f'{player.name} with id {i}')
        f.write(f'{player.name} with id {i}\n')
    
    pred_times_sim = []
    pred_times_adv = []
    pred_times_full = []

    win_rates = np.zeros(4)
    bar = tqdm(range(game_count))
    for i in bar:
        random.shuffle(players)
        game = LudoGame(players)
        winner = players[game.play_full_game()]
        win_rates[tournament_player_ids[winner]] += 1

        bar.set_description(f'Win rates {np.around(win_rates/np.sum(win_rates)*100, decimals=2)}')


    win_rates = win_rates / game_count
    print(f'Win rates {np.around(win_rates/np.sum(win_rates)*100, decimals=4)}')
    f.write(f'Win rates {np.around(win_rates/np.sum(win_rates)*100, decimals=4)}\n')

    for player in tournament_player_ids:
        if player.name == 'random':
            continue
        mean_time = np.mean(player.pred_time)
        print(f'Mean prediction time of {player.name}: {mean_time:.6f}')
        f.write(f'Mean prediction time of {player.name}: {mean_time:.6f}\n')

    f.close()