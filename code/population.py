from organism import Organism
import numpy as np
from numpy.random import random
from bisect import bisect

class Population:
    '''
    The population class. Encodes the following information.
    1. The size of the population
    2. The elitism ratio
    3. The mutation ratio
    '''

    def __init__(self, size=10, mutation=0.3):
        Organism.mutation = mutation
        self.pop_size = size
        self.mutation = mutation
        self.children_per_gen = self.pop_size/2
        self.population = []

    def chooseParents(self):
        cumulative_fitness = np.cumsum([org.fitness for org in self.population])
        fitness_sum = cumulative_fitness[-1]
        self.parents1 = sorted([self.population[bisect(cumulative_fitness, fitness_sum*random())] \
          for _ in xrange(self.children_per_gen)], key=lambda org: org.fitness, reverse=True)
        self.parents2 = sorted([self.population[bisect(cumulative_fitness, fitness_sum*random())] \
          for _ in xrange(self.children_per_gen)], key=lambda org: org.fitness, reverse=True)

    def reproduce(self):
        self.children = [Organism.reproduce(p1, p2) for (p1, p2) in zip(self.parents1, self.parents2)]
        self.population += self.children
        self.population = sorted(self.population, key=lambda org: org.fitness, reverse=True)[:self.pop_size]

    def create_next_gen(self):
        self.chooseParents()
        self.reproduce()

    @staticmethod
    def init_random(size=10, mutation=0.3, solve_method="logistic"):
        new_pop = Population(size, mutation)
        new_pop.population = [Organism.init_random(np.random.randint(5,32), solve_method) for _ in xrange(size)]
        return new_pop
