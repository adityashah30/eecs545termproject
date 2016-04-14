
# -*- coding: utf-8 -*-
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
        sorted(weighted_population,key=itemgetter(1));
        leng=len(weighted_population);
        eli=(math.floor((self.elitism)*self.pop_size))
        new_weightedpop=np.asarray(weighted_population);
        
        
#        for i in range(math.floor((self.elitism)*leng)):
#            new_population.append(weighted_population)
        new_population=new_weightedpop[0:eli,0].tolist();
        children=[];
        print (self.pop_size)
        for _ in range(int(self.pop_size)):
            ind1 = self.weighted_choice(weighted_population)
            ind2 = self.weighted_choice(weighted_population)
            child = ind1.reproduce(ind2)
            
            
            if (random.uniform(0,1)<=self.mutation):
                child.mutate();
            fitness_val = child.fitness_measure()
            children.append((child,fitness_val))
            
            
        sorted(children,key=itemgetter(1),reverse=True);
        leng=len(children);
        eli_new=(math.floor((1-self.elitism)*self.pop_size))
        new_children=np.asarray(children)
        final_children=new_children[0:eli_new,0].tolist();
            
        new_population=new_population+final_children
        
        new_gen.pop_size=len(new_population)
        new_gen.elitism=self.elitism
        new_gen.mutation=self.mutation
        new_gen.new_population=new_population
        return new_gen
        
         
    

            
    
                
                
            
    