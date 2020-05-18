"""Recombinators
"""


import numpy as np


class BaseRecombinator:
    """BaseRecombinator
    """
    name = 'base'
    args = None

    def recombine(self, parent1: np.ndarray, parent2: np.ndarray):
        pass

    @staticmethod
    def new_gamma(size: int, alpha: float):
        return np.random.uniform(-alpha, 1+alpha, size)

    def __call__(self, *args):
        return self.recombine(*args)


class NoneRecombinator(BaseRecombinator):
    """NoneRecombinator
    """
    name = 'None'
    args = []

    def __init__(self, gene_count: int):
        pass

    def recombine(self, parent1: np.ndarray, parent2: np.ndarray):
        return parent1, parent2


class UniformRecombinator(BaseRecombinator):
    """Real uniform recombinator
    """
    name = 'Uniform'
    args = []

    def __init__(self, gene_count: int):
        pass

    def recombine(self, parent1: np.ndarray, parent2: np.ndarray):
        """Combines parent 1 and parent 2 into two children.
        
        Arguments:
            parent1 {np.ndarray}
            parent2 {np.ndarray}
        
        Returns:
            np.ndarray -- children (1 and 2) of parent 1 and parent 2
        """
        mask = np.random.uniform(0, 1, parent1.size) < 0.5
        child1, child2 = parent1.copy(), parent2.copy()
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]
        return child1, child2


class WholeRecombinator(BaseRecombinator):
    """Real whole arithmetic recombinator
    """
    name = 'Whole'
    args = []

    def __init__(self, gene_count):
        """Creates a WholeRecombinator object
        
        Arguments:
            gene_count {int} -- number of genes
        """
        self.blend_combinator = BlendRecombinator(gene_count, alpha=0.0)
    
    def recombine(self, *parents):
        """Recombines parents into childs
        
        Returns:
            np.ndarray -- child 1 and 2 of parents
        """
        return self.blend_combinator.recombine(*parents)


class BlendRecombinator(BaseRecombinator):
    """Real blend recombinator
    """
    name = 'Blend'
    args = [('alpha', float)]

    def __init__(self, gene_count: int, alpha: float):
        """Creates a BlendRecombinator object
        
        Arguments:
            gene_count {int} --
            alpha {float} --
        """
        self.gene_count = gene_count
        self.alpha = alpha

    def recombine(self, parent1: np.ndarray, parent2: np.ndarray):
        """Recombines parent 1 and 2 into child 1 and 2.
        
        Arguments:
            parent1 {np.ndarray} --
            parent2 {np.ndarray} --
        
        Returns:
            np.ndarray -- child 1 and 2 of parent 1 and 2
        """
        gamma1, gamma2 = (self.new_gamma(parent1.size, self.alpha) for _ in range(2))
        child1 = gamma1 * parent1 + (1 - gamma1) * parent2
        child2 = gamma2 * parent2 + (1 - gamma2) * parent1
        return child1, child2


def get_recombinator(name):
    """Get recombinator with name, fx 'none', 'uniform', 'whole' and 'blend'.
    
    Arguments:
        name {string} -- name of the recombinator
    
    Returns:
        object -- Recombinator
    """
    recombinators = [NoneRecombinator, UniformRecombinator, WholeRecombinator, BlendRecombinator]
    recombinator_map = {}
    for recombinator in recombinators:
        recombinator_map[recombinator.name] = recombinator
    return recombinator_map[name]