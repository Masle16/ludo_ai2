"""Plot genes
"""

import os
import sys
import argparse
import numpy as np
import glob
from matplotlib import pyplot as plt
from ga_players import get_ga_player


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='genetic_algorithm/populations/Simple/Simple_Normal-100-10_None_OneStep', help='')
args = parser.parse_args()


def get_player_class(folder_path):
    """Get player class from folder
    
    Arguments:
        folder_path {string} -- path/to/folder
    
    Returns:
        class, list -- the player object and its args
    """
    player_name, *player_args = os.path.basename(folder_path).split('_')[0].split('-')
    Player = get_ga_player(player_name)
    player_args = [arg_def[1](arg) for arg, arg_def in zip(player_args, Player.args)]
    return Player, player_args


def load_populations(folder_path):
    """Load populations from folder path
    
    Arguments:
        folder_path {string} -- path/to/folder
    
    Returns:
        np.array -- generation ids, genes and sigmas
    """
    assert os.path.isdir(folder_path), f'no folder found: {folder_path}'
    population_paths = glob.glob(folder_path + '/*.pop.npy')
    generation_ids = np.array([int(os.path.basename(path).split('.')[0]) for path in population_paths])
    populations = np.array([np.load(f) for f in population_paths])
    idx = np.argsort(generation_ids)
    generation_ids = generation_ids[idx]
    populations = populations[idx]
    Player, _ =  get_player_class(folder_path)
    gene_count = Player.gene_count
    genes = populations[:, :, :gene_count]
    sigmas = populations[:, :, gene_count:]
    return generation_ids, genes, sigmas


def plot(gene_idx, flat_genes, xlabel, ylabel):
    """Plot
    
    Arguments:
        gene_idx {(int, int)} -- gene index
        flat_genes {np.array} -- numpy array of genes
    """
    plt.figure(f'{gene_idx[0]}/{gene_idx[1]}', figsize=(3,3))
    c = np.arange(len(flat_genes))
    all_genes = flat_genes[:, gene_idx]
    xs = all_genes[:, 0]
    ys = all_genes[:, 1]
    plt.scatter(xs, ys, c=c, cmap='brg', s=8)
    ax = plt.gca()
    ax.tick_params(length=0)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    plt.grid()
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    # plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def main():
    generation_ids, genes, sigmas = load_populations(args.path)
    flat_genes = genes.reshape((-1, genes.shape[-1]))
    plot((0,1), flat_genes, r'$\theta_1$', r'$\theta_2$')
    plot((2,3), flat_genes, r'$\theta_3$', r'$\theta_4$')
    plt.show()

if __name__ == '__main__':
    main()
