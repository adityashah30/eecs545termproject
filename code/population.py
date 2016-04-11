from organism import Organism

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
        self.population = [Organism.init_random() for _ in xrange(size)]

    def create_next_gen(self):
	 for organism in population:
       fitness_val = fitness(individual)

      # Generate the (individual,fitness) pair, taking in account whether or
      # not we will accidently divide by zero.
      if fitness_val == 0:
        pair = (individual, 1.0)
      else:
        pair = (individual, 1.0/fitness_val)

      weighted_population.append(pair)

    population = []

    # Select two random individuals, based on their fitness probabilites, cross
    # their genes over at a random point, mutate them, and add them back to the
    # population for the next iteration.
    for _ in xrange(POP_SIZE/2):
      # Selection
      ind1 = weighted_choice(weighted_population)
      ind2 = weighted_choice(weighted_population)

      # Crossover
      ind1, ind2 = crossover(ind1, ind2)

      # Mutate and add back into the population.
      population.append(mutate(ind1))
      population.append(mutate(ind2))

  # Display the highest-ranked string after all generations have been iterated
  # over. This will be the closest string to the OPTIMAL string, meaning it
  # will have the smallest fitness value. Finally, exit the program.
  fittest_string = population[0]
  minimum_fitness = fitness(population[0])
		
	   
	   
		
