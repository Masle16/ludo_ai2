"""Mutators
"""


import numpy as np
import math


class BaseMutator:
    """Base Mutator
    """
    name = 'Base'
    args = []
    gene_count = None
    chromosome_length = None

    def mutate(self, chromosome: np.ndarray):
        """Mutate
        
        Arguments:
            chromosome {np.ndarray} -- [description]
        """
        pass

    def __call__(self, *args):
        """Function call for mutate
        """
        return self.mutate(*args)


class NoneMutator(BaseMutator):
    """None mutator
    """
    name = 'None'
    args = []

    def __init__(self, gene_count):
        self.gene_count = self.chromosome_length = gene_count
        pass

    def mutate(self, chromosome: np.ndarray):
        return chromosome


class NormalMutator(BaseMutator):
    """Real normal mutator
    """
    name = 'Normal'
    args = [('sigma', float)]

    def __init__(self, gene_count: int, sigma: float):
        self.gene_count = self.chromosome_length = gene_count
        self.sigma = sigma
        pass

    def mutate(self, chromosome: np.ndarray):
        return chromosome + np.random.randn(self.gene_count) * self.sigma


class OneStepMutator(BaseMutator):
    """Real adaptive one step normal mutator
    """
    name = 'OneStep'
    # args = [('learning_rate', float)]
    args = []

    # def __init__(self, gene_count: int, lr: float):
    def __init__(self, gene_count: int):
        # self.lr = lr
        self.lr = 0.1
        self.gene_count = gene_count
        self.chromosome_length = gene_count + 1
        self.tau = self.lr / math.sqrt(gene_count)
        self.sigma_min = 1e-3

    def mutate(self, chromosome: np.ndarray):
        genes, sigma = chromosome[:-1], chromosome[-1]
        _sigma = max(self.sigma_min, sigma * pow(math.e, self.tau * np.random.randn()))
        _genes = chromosome[:-1] + np.random.randn(chromosome.size-1) * _sigma
        return np.concatenate((_genes, [_sigma]))


class NStepMutator(BaseMutator):
    """Real adaptive N step normal mutator
    """
    name = 'NStep'
    # args = [('learning_rate_global', float), ('learning_rate_local', float)]
    args = []

    # def __init__(self, gene_count: int, lr_global: float, lr_local: float):
    def __init__(self, gene_count: int):
        """Creates a NStepMutator object
        
        Arguments:
            gene_count {int} -- number of genes
            lr_global {float} -- learning rate global
            lr_local {float} -- learning rate local
        """
        # self.lr_global = lr_global
        # self.lr_local = lr_local
        self.lr_global = 0.1
        self.lr_local = 0.1
        self.gene_count = gene_count
        self.chromosome_length = gene_count * 2
        self.tau_global = self.lr_global / math.sqrt(2 * gene_count)
        self.tau_local = self.lr_local / math.sqrt(2 * math.sqrt(gene_count))
        self.sigma_min = 1e-3

    def mutate(self, chromosome: np.ndarray):
        genes, sigma = chromosome[:self.gene_count], chromosome[self.gene_count:]
        sig_exp = self.tau_local * np.random.randn(self.gene_count) + self.tau_global * np.random.randn()
        _sigma = np.maximum(self.sigma_min, sigma * np.exp(sig_exp))
        _genes = genes + _sigma * np.random.randn(self.gene_count)
        return np.concatenate((_genes, _sigma))


def get_mutator(name):
    """Get mutator with a specific name.
    
    Arguments:
        name {string} -- name of the mutator, fx 'none', 'normal', 'one step' or 'n step'
    
    Returns:
        object -- mutator
    """
    mutators = [NoneMutator, NormalMutator, OneStepMutator, NStepMutator]
    mutator_map = {}
    for mutator in mutators:
        mutator_map[mutator.name] = mutator
    return mutator_map[name]
