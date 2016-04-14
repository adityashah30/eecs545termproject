"""
Created on Tue Apr 12 21:27:18 2016
@author: SHARATH NS
"""

from organism import Organism
import random
from operator import itemgetter
import math
import numpy as np


class Population:
    '''
    The population class. Encodes the following information.
    1. The size of the population
    2. The elitism ratio
    3. The mutation ratio
    '''

    def __init__(self, size=10, elitism=0.15, mutation=0.3):
        print size, elitism, mutation
        self.pop_size = size
        self.elitism = elitism
        self.mutation = mutation
        self.population = []

    def create_next_gen(self):
        new_gen=Population(self.pop_size, self.elitism, self.mutation)
        self.population = sorted(self.population, key=lambda a: a.fitness, reverse=True)
        elite_pop_size = int(self.elitism*self.pop_size)
        elite_population = self.population[:elite_pop_size]
        new_population = []
        for _ in xrange(self.pop_size):
            parent1, parent2 = np.random.choice(self.population, 2, replace=False)
            child1 = parent1.reproduce(parent2)
            if np.random.random() <= self.mutation:
                child1.mutate()
            new_population.append(child1)
            child2 = parent2.reproduce(parent1)
            if np.random.random() <= self.mutation:
                child2.mutate()
            new_population.append(child2)
        new_population = sorted(new_population, key=lambda a: a.fitness, reverse=True)
        new_population = new_population[:self.pop_size-elite_pop_size]
        new_population += elite_population
        new_gen.population = new_population
        return new_gen

    @staticmethod
    def init_random(size=10, elitism=0.15, mutation=0.3):
        new_pop = Population(size, elitism, mutation)
        new_pop.population = [Organism.init_random(15) for _ in range(size)]
        return new_pop
