"""Selections
"""


import math
import random
import numpy as np
import copy
from tqdm import tqdm
from pyludo.LudoGame import LudoGame


class BaseTournament:
    """Base tournament class
    """
    name = 'Base'
    args = []
    population = np.array([])
    progress_bar = None
    cur_tournament_count = None
    cur_generation = 0
    total_game_count = 0

    # def __init__(self, Player, pop_size, pop_init, recombine, mutate):
    def __init__(self, Player, pop_size, recombine, mutate):
        """Creates a BaseTournament object
        
        Arguments:
            Player {object} -- player object
            pop_size {int} -- population size
            pop_init {function} -- function for population initialization
            recombine {object} -- recombinator object
            mutate {object} -- mutator object
        """
        self.Player = Player
        self.mutate = mutate
        self.recombine = recombine
        # self.population = pop_init(pop_size)
        self.population = self.pop_init(pop_size)
        for chromosome in self.get_flat_pop():
            chromosome[:] = Player.normalize(chromosome)
        self.population_size = pop_size
        self.tournaments_per_generation = pop_size // 4
    
    def pop_init(self, pop_size):
        """Initialize population
        
        Arguments:
            sigma_count {int} -- [description]
            pop_size {int} -- population size
        """
        sigma_count = self.mutate.chromosome_length - self.Player.gene_count
        genes = np.random.randn(pop_size * self.Player.gene_count).reshape((pop_size, -1))
        sigma = np.exp(np.random.randn(pop_size * sigma_count) - 2).reshape((pop_size, -1))
        return np.concatenate((genes, sigma), axis=1)

    def get_flat_pop(self):
        """Get flatten population
        
        Returns:
            np.array -- flatten population
        """
        return self.population.reshape((-1, self.population.shape[-1]))

    def play_tournament(self, chromosome_ids, game_count: int):
        """Play tournament
        
        Arguments:
            chromosome_ids {np.array} -- ids of the chromosomes to play
            game_count {int} -- number of games
        """
        flat_pop = self.get_flat_pop()
        chromosomes = flat_pop[chromosome_ids]
        players = [self.Player(chromosome) for chromosome in chromosomes]
        
        tournament_player_ids = {}
        for i, player in enumerate(players):
            tournament_player_ids[player] = i
        
        win_rates = np.zeros(4)
        # bar = tqdm(range(game_count))
        # for _ in bar:
        for _ in range(game_count):
            random.shuffle(players)
            game = LudoGame(players)
            winner = players[game.play_full_game()]
            win_rates[tournament_player_ids[winner]] += 1

            # bar.set_description(f'Win rates {np.around(win_rates/np.sum(win_rates)*100, decimals=2)}')
        
        ranked_chromosome_ids = chromosome_ids[np.argsort(-win_rates)]
        children = self.recombine(*flat_pop[ranked_chromosome_ids[:2]])
        children = [self.mutate(child) for child in children]
        children = [self.Player.normalize(child) for child in children]
        flat_pop[ranked_chromosome_ids[2:]] = children

        self.total_game_count += game_count
        self.cur_tournament_count += 1

    def step(self, generation_count=1):
        total_tournament_count = generation_count * self.tournaments_per_generation
        self.cur_tournament_count = 0

        for _ in range(generation_count):
            # print(f'Generation {self.cur_generation}')
            self.next_generation()
            self.cur_generation += 1
    
    def next_generation(self):
        pass


class Tournament(BaseTournament):
    """Tournament
    """
    name = 'Normal'
    args = [('pop_size', int), ('games_per_tournament', int)]

    # def __init__(self, Player, pop_init, recombine, mutate, pop_size, games_per_tournament):
    def __init__(self, Player, recombine, mutate, pop_size, games_per_tournament):
        """Creates a Tournament object
        
        Arguments:
            Player {object} -- Player object
            pop_init {function} -- population initialization
            recombine {object} -- Recombinator object
            mutate {object} -- Mutator object
            pop_size {int} -- population size
            games_per_tournament {int} -- games per tournament
        """
        # super(Tournament, self).__init__(Player, pop_size, pop_init, recombine, mutate)
        super(Tournament, self).__init__(Player, pop_size, recombine, mutate)
        self.games_per_tournament = games_per_tournament
        self.all_chromosome_ids = np.arange(pop_size)
    
    def next_generation(self):
        """Next tournament
        """
        np.random.shuffle(self.all_chromosome_ids)
        for tournament_id in range(self.population_size//4):
            chromosome_ids = self.all_chromosome_ids[tournament_id * 4:tournament_id * 4 + 4]
            self.play_tournament(chromosome_ids, self.games_per_tournament)


class CellularTournamenet(BaseTournament):
    """CellularTournament
    """
    name = 'Cellular'
    args = [('pop_size', int), ('games_per_tournament', int)]

    # def __init__(self, Player, pop_init, recombine, mutate, population_size, games_per_tournament):
    def __init__(self, Player, recombine, mutate, population_size, games_per_tournament):
        """Creates a CellularTournament object

        Arguments:
            Player {object} -- player object
            pop_init {function} -- function to generate population
            recombine {object} -- recombinator object
            mutate {object} -- mutator object
            population_size {int} -- population size
            games_per_tournament {int} -- games per tournament
        """
        grid_size = int(round(math.sqrt(population_size)))
        assert population_size % grid_size == 0
        assert grid_size % 2 == 0
        # super(CellularTournamenet, self).__init__(Player, population_size, pop_init, recombine, mutate)
        super(CellularTournamenet, self).__init__(Player, population_size, recombine, mutate)
        self.population = self.population.reshape((grid_size, grid_size, -1))
        # self.population = self.population.reshape((grid_size+1, grid_size, -1))
        self.grid_size = grid_size
        self.games_per_tournament = games_per_tournament
    
    def next_generation(self, generation_count=1):
        """Next generation
        
        Keyword Arguments:
            generation_count {int} -- generation count (default: {1})
        """
        off_x, off_y = [(0, 0), (0, 1), (1, 1), (1, 0)][self.cur_generation % 4]
        for x in range(0, self.grid_size, 2):
            for y in range(0, self.grid_size, 2):
                chromosome_ids = np.array([
                    ((y + dy + off_y) % self.grid_size) * self.grid_size + ((x + dx + off_x) % self.grid_size)
                    for dx, dy in ((0, 0), (0, 1), (1, 1), (1, 0))
                ])
                self.play_tournament(chromosome_ids, self.games_per_tournament)
        # off_x, off_y = [(0, 0), (0, 1), (1, 1), (1, 0)][self.cur_generation % 4]
        # for x in range(0, self.grid_size, 2):
        #     for y in range(0, self.grid_size, 2):
        #         chromosome_ids = np.array([
        #             ((y + dy + off_y) % self.grid_size) * self.population.shape[0] + ((x + dx + off_x) % self.grid_size)
        #             for dx, dy in ((0, 0), (0, 1), (1, 1), (1, 0))
        #         ])
        #         self.play_tournament(chromosome_ids, self.games_per_tournament)


class IslandTournament(BaseTournament):
    name = 'Island'
    args = [
        ('island_count', int),
        ('chromosomes_per_island', int),
        ('generations_per_epoch', int),
        ('migration_count', int),
        ('games_per_tournament', int)
    ]

    def __init__(self, Player, recombine, mutate, island_count: int,
                 chromosomes_per_island: int, generations_per_epoch: int,
                 migration_count: int, games_per_tournament: int):
        """Creates a IslandTournament object
        
        Arguments:
            Player {object} -- player object
            pop_init {function} -- population initilization
            recombine {object} -- recombinator object
            mutate {object} -- mutator object
            island_count {int} -- island count
            chromosomes_per_island {int} -- chromosomes per island
            generations_per_epoch {int} -- number of generation per epoch
            migration_count {int} -- migration count
            games_per_tournament {int} -- games per tournament
        """
        assert (chromosomes_per_island % 4 == 0)
        super(IslandTournament, self).__init__(Player, island_count * chromosomes_per_island, recombine, mutate)
        self.island_count = island_count
        self.chromosome_per_island = chromosomes_per_island
        self.population = self.population.reshape((island_count, chromosomes_per_island, -1))
        self.migration_count = migration_count
        self.generations_per_epoch = generations_per_epoch
        self.games_per_tournament = games_per_tournament
        self.all_island_chromosome_ids = np.arange(chromosomes_per_island)

    def next_generation(self):
        """Performs evaluation and selection
        """
        for island_id in range(self.island_count):
            np.random.shuffle(self.all_island_chromosome_ids)
            for tournament_id in range(self.chromosome_per_island // 4):
                chromosome_ids = self.all_island_chromosome_ids[tournament_id * 4:tournament_id * 4 + 4].copy()
                chromosome_ids += island_id * self.chromosome_per_island
                self.play_tournament(chromosome_ids, self.games_per_tournament)
    
        if self.cur_generation % self.generations_per_epoch == 0:
            self.migrate()
    
    def migrate(self):
        """Picks self.migration_count on each island and shuffle those chromosomes
        """
        migrant_ids = np.empty((self.island_count, self.migration_count), dtype=np.int)
        for island_id in range(self.island_count):
            migrant_ids[island_id] = np.random.choice(self.all_island_chromosome_ids, self.migration_count, replace=False)
        old_migrant_ids = migrant_ids.reshape(-1)
        new_migrant_ids = old_migrant_ids.copy()
        np.random.shuffle(new_migrant_ids)
        self.get_flat_pop()[new_migrant_ids] = self.get_flat_pop()[old_migrant_ids]


def get_selection(name):
    """Get the selection with name, fx 'tournament', 'cellular tournament' or 
    'island tournament'
    
    Arguments:
        name {string} -- the selection class name
    
    Returns:
        object -- the selection object with name
    """
    selections = [Tournament, CellularTournamenet, IslandTournament]
    selector_map = {}
    for selection in selections:
        selector_map[selection.name] = selection
    return selector_map[name]