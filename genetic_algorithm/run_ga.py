"""Run genetic algorithm
"""


import os
import argparse
import glob
import functools
import sys
import numpy as np
import time
from tqdm import tqdm
from selections import get_selection
from recombinators import get_recombinator
from mutators import get_mutator
from ga_players import get_ga_player


parser = argparse.ArgumentParser()
parser.add_argument('--player', nargs='+', default=['Advanced'])
# parser.add_argument('--selection', nargs='+', default=['Island', 'island_count=5', 'chromosomes_per_island=20', 'generations_per_epoch=1', 'migration_count=1', 'games_per_tournament=10'])
parser.add_argument('--selection', nargs='+', default=['Normal', 'pop_size=100', 'games_per_tournament=10'])
parser.add_argument('--recombination', nargs='+', default=['Uniform'])
parser.add_argument('--mutation', nargs='+', default=['Normal', 'sigma=0.1'])
parser.add_argument('--gen_count', type=int, default=1000)
parser.add_argument('--save_nth_gen', type=int, default=10)
parser.add_argument('--cont', action='store_const', const=True, default=True)
args = parser.parse_args()


def parse_args(arguments, required_args):
    """Parse arguments to objects (Player, Selection, Recombination and Mutator)
    
    Arguments:
        args {list} -- list of input argument, fx (pop_size=10)
        required_args {list} -- list of required argument and type
    
    Returns:
        list -- parsed_args is a list with types and str_args is a list of strings
    """
    assert len(arguments) == len(required_args), f'Expected args: {required_args}'
    parsed_args, str_args = [], []
    for arg_i, (arg_name, typ) in enumerate(required_args):
        provided_arg_name, val = arguments[arg_i].split('=')
        assert provided_arg_name == arg_name, f'Expected "{arg_name}", got "{provided_arg_name}"'
        parsed_args.append(typ(val))
        str_args.append(val)
    return parsed_args, str_args

def args_str_to_string(args_str):
    """Convert args_str to a string
    
    Arguments:
        args_str {list} -- list of arguments
    
    Returns:
        string -- string of arguments
    """
    if args_str:
        return '-' + '-'.join(args_str)
    return ''

def save(folder_path, gen_id, population):
    """Save population to folder as npy file
    
    Arguments:
        folder_path {string} -- path/to/folder
        gen_id {int} -- generation id
        population {np.array}
    """
    file_writing_name = folder_path + f'/{gen_id}.pop.writing.npy'
    file_writtin_name = folder_path + f'/{gen_id}.pop.npy'
    np.save(file_writing_name, population)
    os.rename(file_writing_name, file_writtin_name)


def main():
    Player = get_ga_player(args.player[0])
    player_args, player_args_str = parse_args(args.player[1:], Player.args)
    gene_count = Player.gene_count

    Selection = get_selection(args.selection[0])
    selection_args, selection_args_str = parse_args(args.selection[1:], Selection.args)

    Recombinator = get_recombinator(args.recombination[0])
    recombinator_args, recombinator_args_str = parse_args(args.recombination[1:], Recombinator.args)

    Mutator = get_mutator(args.mutation[0])
    mutator_args, mutator_args_str = parse_args(args.mutation[1:], Mutator.args)

    mutator = Mutator(gene_count, *mutator_args)
    recombinator = Recombinator(gene_count, *recombinator_args)

    generation_count = args.gen_count
    save_every_nth_generation = args.save_nth_gen

    # pop_init = functools.partial(Player.pop_init, Player, mutator.chromosome_length - Player.gene_count)
    # selection = Selection(Player, pop_init, recombinator, mutator, *selection_args)
    selection = Selection(Player, recombinator, mutator, *selection_args)

    folder_name = f'{Player.name}{args_str_to_string(player_args_str)}_' \
                  f'{Selection.name}{args_str_to_string(selection_args_str)}_' \
                  f'{Recombinator.name}{args_str_to_string(recombinator_args_str)}_' \
                  f'{Mutator.name}{args_str_to_string(mutator_args_str)}'
    folder_path = f'genetic_algorithm/populations/{Player.name}/{folder_name}'

    if args.cont:
        tmp = ''
    else:
        tmp = ' not'
    assert os.path.isdir(folder_path) == args.cont, f'{folder_path} should{tmp} exist'
    
    if not args.cont:
        os.mkdir(folder_path)
    else:
        gen_ids = [int(os.path.basename(path).split('.')[0]) for path in glob.glob(folder_path+'/*.pop.npy')]
        selection.cur_generation = max(gen_ids)
        selection.population = np.load(folder_path + f'/{selection.cur_generation}.pop.npy')

    if generation_count == 0:
        generation_count = int(1e9)

    if not args.cont:
        save(folder_path, 0, selection.get_flat_pop())

    t1 = time.time()
    start_generation = selection.cur_generation

    bar = tqdm(range(selection.cur_generation, generation_count))
    for i in bar:
        selection.step()
        
        if selection.cur_generation % save_every_nth_generation == 0:
            save(folder_path, selection.cur_generation, selection.get_flat_pop())
        
        # flat_pop = selection.get_flat_pop()
        # chromo_mean = flat_pop.mean(axis=0)
        # chromo_std = flat_pop.std(axis=0)

        # with np.printoptions(precision=3, suppress=True):
        #     print(f'Gen {i}/{generation_count} Gene: mean {chromo_mean[:gene_count]} std {chromo_std[:gene_count]} Sigma: mean{chromo_mean[gene_count:]} std {chromo_std[gene_count:]}')

    t2 = time.time()
    print(f'From generation {start_generation} to {generation_count} took {(t2-t1):.3f} seconds')


if __name__ == '__main__':
    main()
