"""Reduce population
"""

import os
import sys
import argparse
import random
import numpy as np
import time
import multiprocessing as mp
from pyludo.LudoGame import LudoGame
from pyludo.StandardLudoPlayers import LudoPlayerRandom
from ga_players import get_ga_player
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='genetic_algorithm/populations/Simple/Simple_Cellular-20-10_Blend-0.5_NStep/1000.pop.npy', help='path/to/population/*.pop.npy')
parser.add_argument('--games_per_tournament', type=int, default=2500, help='games per tournament')
parser.add_argument('--process_count', type=int, default=8, help='process count')
args = parser.parse_args()


def tournament(chromosomes):
    """Tournament
    
    Arguments:
        chromosomes {np.array} -- chromosomes
        player {ga_player} -- GA player (simple, advanced or ANN)
        game_count {int} -- game count
    
    Returns:
        int -- id of the winner
    """
    players = [args.player(chromosome) for chromosome in chromosomes]
    
    while len(players) < 4:
        players.append(LudoPlayerRandom)
    
    tournament_player_ids = {}
    for i, player in enumerate(players):
        tournament_player_ids[player] = i

    win_rates = np.zeros(4)
    for i in range(args.games_per_tournament):
        random.shuffle(players)
        game = LudoGame(players)
        winner = players[game.play_full_game()]
        win_rates[tournament_player_ids[winner]] += 1
    
    ranked_player_ids = np.argsort(-win_rates)
    for id in ranked_player_ids:
        if id < len(chromosomes):
            return id


def get_required_tournament_count(pop_size: int, played_tournaments=0):
    """Get required tournament count
    
    Arguments:
        pop_size {int} -- population size
    
    Keyword Arguments:
        played_tournaments {int} -- played tournaments (default: {0})
    
    Returns:
        int -- played tournaments
    """
    if pop_size == 1:
        return played_tournaments

    n = pop_size // 4
    new_pop_size = n + pop_size % 4

    if n == 0:
        n = 1
        new_pop_size = 1
    
    played_tournaments += n

    return get_required_tournament_count(new_pop_size, played_tournaments)


def main():
    folder_path = os.path.dirname(args.path)
    folder_name = os.path.basename(folder_path)
    gen_id = int(os.path.basename(args.path).split('.')[0])

    player_name = folder_name.split('_')[0]
    player = get_ga_player(player_name)
    args.player = player

    population = list(np.load(args.path))
    n = len(population)

    required_tournament_count = get_required_tournament_count(n)
    required_game_count = required_tournament_count * args.games_per_tournament
    print(f'Required game count: {required_game_count}')

    bar = tqdm(total=required_tournament_count-1)

    tournaments_played = 0
    while n > 1:
        tournament_count = n // 4
        if tournament_count == 0:
            tournament_count = 1
        
        # print(f'Currently {n} players in the population. Playing {tournament_count} tournaments.')
        # bar.write(f'Currently {n} players in the population. Playing {tournament_count} tournaments.')
        bar.set_description(f'Players {n} Tournaments {tournament_count}')

        next_population = population[tournament_count * 4:]

        chromosomes = [population[i*4:(i+1)*4] for i in range(tournament_count)]
        indices = [i*4 for i in range(tournament_count)]

        p = mp.Pool(processes=args.process_count)
        winner_ids = p.map(tournament, [chromosome for chromosome in chromosomes])
        p.close()
        p.join()

        for chromosome, winner_id in zip(chromosomes, winner_ids):
            winner = chromosome[winner_id]
            next_population.append(winner)

        population = next_population
        n = len(population)

        bar.update(args.process_count)

    bar.close()
    
    winner = population[0]
    np.save(f'{folder_path}/{gen_id}.pop.winner.npy', winner)

    print(f'Saved winner to {folder_path}/{gen_id}.pop.winner.npy')


if __name__ == '__main__':
    main()
