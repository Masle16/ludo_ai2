"""[summary]
"""


import os
import numpy as np
import glob
from pyludo.StandardLudoPlayers import *
from ga_players import get_ga_player


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

def get_opps_class(opp_name):
    """Get the opponents class
    
    Arguments:
        opp_name {string} -- name of opponent class
    
    Returns:
        object -- player class with opponent name
    """
    opps = [
        LudoPlayerRandom,
        LudoPlayerDefensive,
        LudoPlayerFast,
        LudoPlayerAggressive,
        SmartPlayer
    ]
    opp_map = {}
    for opp in opps:
        opp_map[opp.name] = opp
    return opp_map[opp_name]

def load_scores(folder_path):
    """load scores from folder
    
    Arguments:
        folder_path {string} -- path/to/folder
    
    Returns:
        np.array -- 
    """
    assert os.path.isdir(folder_path), f'no folder found: {folder_path}'
    score_paths = glob.glob(folder_path + '/*.scores.*.npy')
    x = np.array([int(os.path.basename(score_path).split('.')[0]) for score_path in score_paths])
    y = np.array([np.load(f) for f in score_paths])
    idx = np.argsort(x)
    return x[idx], y[idx]

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