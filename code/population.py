# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 21:27:18 2016

@author: SHARATH NS
"""

from organism import Organism
import random

class Population:
    '''
    The population class. Encodes the following information.
    1. The size of the population
    2. The elitism ratio
    3. The mutation ratio

    '''

    def __init__(self, size=100, elitism=0.15, mutation=0.3):
        self.pop_size = size
        self.elitism = elitism
        self.mutation = mutation
        self.population = [Organism.init_random(15) for _ in range(size)]
  
    def weighted_choice(self,items):
 
        weight_total = sum((item[1] for item in items))
        n = random.uniform(0, weight_total)
        for item, weight in items:
          if n < weight:
            return item
            n = n - weight
        return item

    def create_next_gen(self):
        new_gen=Population();
        weighted_population=[]
        
        for Org in self.population:
            fitness_val = Org.fitness_measure()
            if fitness_val == 0:
                pair = (Org, 1.0)
            else:
                pair = (Org, 1.0/fitness_val)
            weighted_population.append(pair)
        
        new_population = [];
        for _ in range(self.pop_size):
            ind1 = self.weighted_choice(weighted_population)
            ind2 = self.weighted_choice(weighted_population)
            child = ind1.reproduce(ind2)
            new_population.append(child.mutate)
        
        new_gen.pop_size=self.pop_size
        new_gen.elitism=self.elitism
        new_gen.mutation=self.mutation
        new_gen.new_population=new_population
        return new_gen
        
         
    

            
    
                
                
            
        
   