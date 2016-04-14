from population import Population

class GA:
    '''
    The Genetic Algorithm (GA) class. Encodes the following information.
    1. The population
    '''

    def __init__(self, gen_count=10, size=10, elitism=0.15, mutation=0.3):
        self.gen_count = gen_count
        self.population = Population.init_random(size, elitism, mutation)

    def search(self):
        for genIter in xrange(self.gen_count):
            print "In generation: ", genIter
            self.population = self.population.create_next_gen()
        return self.population.population
