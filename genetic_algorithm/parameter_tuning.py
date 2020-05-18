"""Parameter tunning for GA
"""


import os
import sys
import multiprocessing as mp
import argparse
import numpy as np
import time
import random
import functools
from glob import glob
# from tqdm import tqdm
from p_tqdm import p_map
from pyludo.LudoGame import LudoGame
from pyludo.StandardLudoPlayers import LudoPlayerRandom
from selections import get_selection
from recombinators import get_recombinator
from mutators import get_mutator
from ga_players import get_ga_player

selections_list = ['Normal', 'Cellular', 'Island']
recombinators_list = ['None', 'Uniform', 'Whole', 'Blend']
mutators_list = ['None', 'Normal', 'OneStep', 'NStep']

parser = argparse.ArgumentParser(description='Random parameter tuning for GA')
parser.add_argument('--player', type=str, default='Advanced', help='player type')
parser.add_argument('--iterations', type=int, default=100, help='number of iteration to run')
parser.add_argument('--gen_count', type=int, default=50, help='number of generation')
parser.add_argument('--games_per_tournament', type=int, default=10, help='number of games per tournament')
parser.add_argument('--process_count', type=int, default=8, help='number of porcesses')
args = parser.parse_args()


def create_player(Player, games_per_tournament=10):
    gene_count = Player.gene_count

    # selection
    idx = np.random.randint(low=0, high=len(selections_list))
    selection_name = selections_list[idx]
    # selection_name = 'Normal'
    Selection = get_selection(selection_name)

    selection_args = []
    if selection_name == 'Normal' or selection_name == 'Cellular':
        selection_args.append(20)
        selection_args.append(games_per_tournament)

    elif selection_name == 'Island':
        selection_args.append(5)
        selection_args.append(4)
        selection_args.append(1)
        selection_args.append(1)
        selection_args.append(games_per_tournament)

    else:
        print(f'the selection is not known, got {selection_name} to index {idx}')
    
    # recombinator
    idx = np.random.randint(low=0, high=len(recombinators_list))
    recombinator_name = recombinators_list[idx]
    # recombinator_name = 'None'
    Recombinator = get_recombinator(recombinator_name)

    recombinator_args = []
    if recombinator_name == 'Blend':
        # alpha = np.random.random_sample()
        alpha = 0.5
        recombinator_args.append(alpha)

    recombinator = Recombinator(gene_count, *recombinator_args)

    # mutator
    idx = np.random.randint(low=0, high=len(mutators_list))
    mutator_name = mutators_list[idx]
    # mutator_name = 'Normal'
    Mutator = get_mutator(mutator_name)

    mutator_args = []
    if mutator_name == 'Normal':
        sigma = np.random.random_sample()
        mutator_args.append(sigma)

    mutator = Mutator(gene_count, *mutator_args)
    recombinator = Recombinator(gene_count, *recombinator_args)

    # pop_init = functools.partial(Player.pop_init, Player, mutator.chromosome_length - Player.gene_count)
    # selection = Selection(Player, pop_init, recombinator, mutator, *selection_args)
    selection = Selection(Player, recombinator, mutator, *selection_args)

    return selection


def tournament(chromosomes, player, game_count=10):
    players = [player(chromosome) for chromosome in chromosomes]

    while len(players) < 4:
        players.append(LudoPlayerRandom)
    
    tournament_player_ids = {}
    for i, player in enumerate(players):
        tournament_player_ids[player] = i

    win_rates = np.zeros(4)

    for i in range(game_count):
        np.random.shuffle(players)
        game = LudoGame(players)
        winner = players[game.play_full_game()]
        win_rates[tournament_player_ids[winner]] += 1
    
    ranked_player_ids = np.argsort(-win_rates)
    for id in ranked_player_ids:
        if id < len(chromosomes):
            return id


def get_required_tournament_count(pop_size, played_tournaments=0):
    if pop_size == 1:
        return played_tournaments
    
    n = pop_size // 4
    new_pop_size = n + pop_size % 4

    if n == 0:
        n = 1
        new_pop_size = 1

    played_tournaments += n

    return get_required_tournament_count(new_pop_size, played_tournaments)


def reduce_pop(player, population, games_per_tournament=10):
    population = list(population)
    n = len(population)

    required_tournament_count = get_required_tournament_count(n)
    required_game_count = required_tournament_count * games_per_tournament

    tournaments_played = 0
    while n > 1:
        tournament_count = n // 4
        if tournament_count == 0:
            tournament_count = 1

        next_population = population[tournament_count * 4:]

        for i in range(tournament_count):
            chromosomes = population[i*4:(i+1)*4]
            winner_id = tournament(chromosomes, player, games_per_tournament)
            next_population.append(chromosomes[winner_id])
            tournaments_played += 1

        population = next_population
        n = len(population)
    
    winner = population[0]
    return winner


def get_player_args(player):
    player_args = ''

    player_args = f'{player.Player.name}'

    player_args += f'_{player.name}'
    if player.name == 'Cellular' or player.name == 'Normal':
        player_args += f'-{player.population_size}-{player.games_per_tournament}'
    else:
        player_args += f'-{player.island_count}-{player.chromosome_per_island}-{player.generations_per_epoch}-{player.migration_count}-{player.games_per_tournament}'

    player_args += f'_{player.recombine.name}'
    if player.recombine.name == 'Blend':
        player_args += f'-{player.recombine.alpha:.2f}'

    player_args += f'_{player.mutate.name}'
    if player.mutate.name == 'Normal':
        player_args += f'-{player.mutate.sigma:.2f}'
    elif player.mutate.name == 'OneStep':
        player_args += f'-{player.mutate.lr:.2f}'
    elif player.mutate.name == 'NStep':
        player_args += f'-{player.mutate.lr_global:.2f}-{player.mutate.lr_local:.2f}'

    return player_args 


def eval_player(player):
    player_args = get_player_args(player)

    for gen in range(args.gen_count):
        player.step()

    population = player.get_flat_pop()
    winner_pop = reduce_pop(args.Player, population, args.games_per_tournament)

    players = [args.Player(winner_pop)]
    while len(players) < 4:
        players.append(LudoPlayerRandom)
    
    tournament_player_ids = {}
    for i, player in enumerate(players):
        tournament_player_ids[player] = i

    win_rates = np.zeros(4)
    game_count = 100
    for i in range(game_count):
        np.random.shuffle(players)
        game = LudoGame(players)
        winner = players[game.play_full_game()]
        win_rates[tournament_player_ids[winner]] += 1
    win_rate = win_rates[0] / game_count

    return [win_rate, player_args, winner_pop]


def main():
    args.Player = get_ga_player(args.player)

    players = [create_player(args.Player, args.games_per_tournament) for _ in range(args.iterations)]
    data = p_map(eval_player, [player for player in players])

    winners = np.array(data)
    idx = np.argsort(-winners[:, 0])
    winners = winners[idx]

    path = f'genetic_algorithm/tuning/{args.Player.name}.txt'
    f = open(f'{path}', 'w')
    for j in range(len(winners)):
        winner = winners[j]
        print(f'Winner No. {j} {winner[1]} with {winner[0]:.2f}')
        f.write(f'Arguments {winner[1]} Win_rate {winner[0]:.2f} Gen_count {args.gen_count}\n')

    f.close()


if __name__ == '__main__':
    main()
