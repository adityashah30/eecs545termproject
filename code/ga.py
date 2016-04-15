from population import Population

class GA:
    '''
    The Genetic Algorithm (GA) class. Encodes the following information.
    1. The population
    '''

    def __init__(self, gen_count=10, size=10, mutation=0.3):
        self.gen_count = gen_count
        self.population = Population.init_random(size, mutation)

    def search(self):
        ofile = "accuracy.txt"
        for genIter in xrange(self.gen_count):
            print "In generation: ", genIter
            self.population.create_next_gen()
            with open(ofile, "a") as fp:
                fp.write(str(genIter)+", "+str(self.population.population[0].fitness)+"\n")
        return self.population.population
