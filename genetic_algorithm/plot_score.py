"""Plot scores
"""


import os
import math
import sys
import argparse
import numpy as np
from glob import glob
from matplotlib import colors as mpl_colors
from matplotlib import pyplot as plt
from collections import defaultdict


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = [mpl_colors.to_rgb(c) for c in prop_cycle.by_key()['color']]


def load_scores(path):
    """Load scores
    
    Arguments:
        path {string} -- path/to/scores.npy
    
    Returns:
        np.array -- matrix with scores
    """
    mat = np.load(path)
    if len(mat.shape) == 2:
        mat = mat[1]
    return mat


def main(args):
    # plt.figure(figsize=(4, 2.5))
    
    training_paths = args.paths
    training_count = len(training_paths)

    opponents_names = args.opponents
    opponent_count = len(opponents_names)

    common_train_args = {0, 1, 2, 3}
    diff_train_args = set()
    
    training_args = [[['p:', 's:', 'r:', 'm:'][i] + str(arg) for i, arg in enumerate(os.path.basename(path).split("_"))]
                     for path in training_paths]

    for train_id in range(1, len(training_args)):
        pre_train_args = training_args[train_id - 1]
        cur_train_args = training_args[train_id]
        assert len(pre_train_args) == len(cur_train_args) == 4, f'Training args does not have a length of 4, got {len(pre_train_args)} and {len(cur_train_args)}'

        for i in common_train_args.copy():
            if pre_train_args[i] != cur_train_args[i]:
                common_train_args.remove(i)
                diff_train_args.add(i)
    
    for train_id, train_path in enumerate(training_paths):
        scores_paths_against_opponent = defaultdict(lambda: [])
        all_scores_paths = glob(train_path + f'/*.scores.*.npy')
        
        selection_str = str(os.path.basename(train_path).split('_')[1]).split('-')[1:3]
        pop_size, games_per_tournament = [int(i) for i in selection_str]
        games_per_generation = pop_size / 4 * games_per_tournament

        for score_path in all_scores_paths:
            opponent_name = os.path.basename(score_path).split('.')[2]
            scores_paths_against_opponent[opponent_name].append(score_path)
        
        for opponent_id, (opponent_name, score_paths) in enumerate(scores_paths_against_opponent.items()):
            Y = np.array([load_scores(f) for f in score_paths])
            X = np.array([int(os.path.basename(score_path).split('.')[0]) for score_path in score_paths])
            
            order = np.argsort(X)
            X, Y = X[order], Y[order]

            if args.match_count:
                X = X * games_per_generation
            
            if args.max_gen:
                mask = X <= args.max_gen
                X, Y = X[mask], Y[mask]
            
            X_all = np.array([[x] * Y.shape[-1] for x in X]).flatten()

            c = colors[(train_id * opponent_count + opponent_id) % len(colors)]
            Y_mean, Y_std, Y_max = Y.mean(axis=1), Y.std(axis=1), Y.max(axis=1)

            label = ' '.join([str(training_args[train_id][arg_i]) for arg_i in diff_train_args])

            if args.scatter:
                plt.scatter(X_all, Y.flatten(), color=(*c, 0.25/math.sqrt(Y.shape[1])), edgecolors='none', marker='s')
            
            if args.mean:
                plt.plot(X, Y_mean, color=c, label=label)

            if args.max:
                plt.plot(X, Y_max, color=c, linestyle='--')

            if args.std:
                plt.fill_between(X, Y_mean-Y_std, Y_mean + Y_std, color=[(*c, 0.15)])

    plt.ylim([0, 1])
    plt.grid()

    if len(diff_train_args) > 0 and not args.no_legend:
        plt.legend()

    if args.tight:
        plt.tight_layout(pad=0.25)

    plt.xlabel('Generation count')
    plt.ylabel('Win rate [%]')
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+', required=True)
    parser.add_argument('--opponents', nargs='+', required=True)
    parser.add_argument('--max_gen', type=int, default=0)
    parser.add_argument('--save', type=str)
    arguments = [
        'mean',
        'std',
        'max',
        'scatter',
        'match_count',
        'no_legend',
        'tight'
    ]
    for arg in arguments:
        parser.add_argument('--'+arg, action='store_const', const=True, default=False)
    args = parser.parse_args()

    main(args)
