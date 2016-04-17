from population import Population
from organism import Organism

class GA:
    '''
    The Genetic Algorithm (GA) class. Encodes the following information.
    1. The population
    '''

    def __init__(self, gen_count=10, size=10, mutation=0.3, solve_method="logistic"):
        self.gen_count = gen_count
        self.population = Population.init_random(size, mutation, solve_method)

    def search(self):
        ofile = "accuracy.txt"
        fp = open(ofile, "w")
        fp.close()
        for genIter in xrange(self.gen_count):
            self.population.create_next_gen()
            best_fitness = self.population.population[0].fitness
            with open(ofile, "a") as fp:
                fp.write(str(genIter)+", "+str(best_fitness)+"\n")
            print "In generation: ", genIter, "; Best fitness: ", best_fitness
        return self.population.population

    @staticmethod
    def full_accuracy(solve_method="logistic", hidden_nodes=50):
        org = Organism(range(Organism.count), hidden_nodes, solve_method)
        return org
        
