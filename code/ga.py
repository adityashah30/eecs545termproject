from population import Population

class GA:
    '''
    The Genetic Algorithm (GA) class. Encodes the following information.
    1. The population
    2. 
    '''

    def __init__(self, gen_count=100, size=100, elitism=0.15, mutation=0.3):
        self.gen_count = gen_count
        self.population = Population(size, elitism, mutation)

    def search(self):
        for _ in xrange(self.gen_count):
            self.population = self.population.create_next_gen()
        return self.population.population
