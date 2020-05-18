"""Eval agent
"""


import argparse
import random
import os
import sys
import numpy as np
import time
# import multiprocessing as mp
from glob import glob
# from tqdm import tqdm
from p_tqdm import p_map
from pyludo.LudoGame import LudoGame
from pyludo.StandardLudoPlayers import *
from ga_players import get_ga_player
from ga_players import GAPlayerSimple, GAPlayerAdvanced, GAPlayerFull
from mikkel.LudoPlayerQLearningAction import LudoPlayerQLearningAction


chromosomes_path = {
    'simple': 'genetic_algorithm/populations/Simple/Simple_Cellular-20-10_Blend-0.5_NStep/1000.pop.winner.npy',
    'advanced': 'genetic_algorithm/populations/Advanced/Advanced_Normal-100-10_Whole_Normal-0.1/1000.pop.winner.npy',
    'full': 'genetic_algorithm/populations/Full/Full_Cellular-100-10_Whole_Normal-0.1/1000.pop.winner.npy'
}


fixed_players = {
    'random': LudoPlayerRandom,
    'smart': SmartPlayer,
    'simple': GAPlayerSimple,
    'advanced': GAPlayerAdvanced,
    'full': GAPlayerFull,
    'mikkel': ''
}


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='')
parser.add_argument('--player', type=str, default='random', help='name of player')
parser.add_argument('--opponent', type=str, default='all', help='name of opponent')
parser.add_argument('--compare', action='store_const', const=True, default=True)
parser.add_argument('--game_count', type=int, default=2500, help='number of games')
parser.add_argument('--process_count', type=int, default=8, help='number of processes')
parser.add_argument('--iterations', type=int, default=1, help='number of iterations of evaluation')
args = parser.parse_args()


def get_player(player_args):
    """Get player
    
    Arguments:
        player_args {list} -- list of player arguments
    
    Returns:
        LudoPlayer -- A genetic algorithm player with chromosome
    """
    len_player_args = len(player_args)
    assert 1 <= len_player_args, f'Need only 1 or 2 players - found: {len_player_args}'
    assert len_player_args <= 2, f'Need only 1 or 2 players - found: {len_player_args}'
    
    if len_player_args == 1:
        if player_args[0] == 'simple' or player_args[0] == 'advanced' or player_args[0] == 'full':
            chromosome = np.load(chromosomes_path[player_args[0]])
            return fixed_players[player_args[0]](chromosome)
        elif player_args[0] == 'mikkel':
            epsilon = 0.05 #
            discount_factor =  0.5 #
            learning_rate = 0.25 # 
            parameters = [epsilon, discount_factor, learning_rate]
            return LudoPlayerQLearningAction(parameters, chosenPolicy="greedy", QtableName='QTable', RewardName='Reward')
        else:
            return fixed_players[player_args[0]]()
    
    Player = get_ga_player(player_args[0])
    chromosome = np.load(player_args[1])
    return Player(chromosome)

def calc_significance(win_rate):
    critical_val = 1.6449
    n = float(args.game_count)

    k = float(win_rate * n)
    z_test = np.abs( ( k - n * 0.5 ) / np.sqrt(n * 0.5 * (1.0 - 0.5)) )

    return z_test > critical_val

def tournament(_players):
    """Holds a tournament for the players
    
    Arguments:
        _players {list} -- list of players
        game_count {int} -- number of games to play
    
    Returns:
        int -- win rates of each player
    """
    assert len(_players) == 4, f'Not enough players for ludo - found: {len(_players)}'

    players = [player for player in _players]
    
    tournament_player_ids = {}
    for i, player in enumerate(players):
        tournament_player_ids[player] = i

    win_rates = np.zeros(4)
    for i in range(args.game_count):
        random.shuffle(players)
        game = LudoGame(players)
        winner = players[game.play_full_game()]
        win_rates[tournament_player_ids[winner]] += 1

    return win_rates / args.game_count


def main():
    if args.player == 'Simple' or args.player == 'Full' or args.player == 'Advanced':
        path = args.path.replace(f'genetic_algorithm/populations/{args.player}/', '')
        path = path.replace('/','-')
        player_name = path.split('_')[0]
        player = get_player([player_name, args.path])
    elif args.player == 'mikkel':
        epsilon = 0.05 #
        discount_factor =  0.5 #
        learning_rate = 0.25 # 
        parameters = [epsilon, discount_factor, learning_rate]
        player = LudoPlayerQLearningAction(parameters, chosenPolicy="greedy", QtableName='QTable', RewardName='Reward')
        player_name = args.player
    else:
        player = fixed_players[args.player]()
        player_name = args.player

    folder = 'genetic_algorithm/agent_evaluations'
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    if args.compare:
        args.dist = (2,2)
    else:
        args.dist = (1,3)

    if args.opponent == 'all':
        opps = [get_player([opp]) for opp in fixed_players]
        players = [[player]*args.dist[0] + [opp]*args.dist[1] for opp in opps]

        data = p_map(tournament, [player for player in players], num_cpus=args.process_count)

        fname = f'score_{args.player}-vs-all_{args.game_count}.txt'
        f = open(f'{folder}/{fname}', 'w')
        f.write(f'Player {player_name} Population {args.path}\n')
        for i, fixed_player in enumerate(fixed_players):
            player_win_rate = np.sum(data[i][:args.dist[1]])
            significance = calc_significance(player_win_rate)
            f.write(f'Opponent {fixed_player} Wins {player_win_rate:.3f} Statistical significance {significance} Games {args.game_count}\n')
        f.close()
    else:
        opponent = get_player(args.opponent)
        opponent_name = '-'.join(args.opponent).replace('/', '-')
        
        players = [player] * dist[0] + [opponent] * dist[1]
        win_rates = tournament(players)
        player_win_rate = np.sum(win_rates[:dist[1]])
        print(f'Player {player_name} Games {args.game_count} Win rate {player_win_rate*100:.4f}')

        fname = f'score_{path}-vs-{opponent_name}_{args.game_count}.txt'
        f = open(f'{folder}/{fname}', 'w')
        f.write(f'Player name --> {player_name}\n')
        f.write(f'Population path --> {args.path}\n')
        f.write(f'Opponent name --> {opponent_name}\n')
        f.write(f'Games --> {args.game_count}\n')
        f.write(f'Player win rate --> {player_win_rate*100:.2f}\n')
        f.close()

if __name__ == '__main__':
    main()
