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
        pass
