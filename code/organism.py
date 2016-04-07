

class Organism:
    '''
    The Organism class. Encodes the following information.
    1. The subset of the features used for classification.
    2. Optional: If the classifier is a neural network, the number of
                 nodes in the hidden layer.
    '''

    def __init__(self, feature_subset=[], hidden_nodes=None):
        self.feature_subset = feature_subset
        if hidden_nodes:
            self.hidden_nodes = hidden_nodes
            self.use_neural_net = True
        else:
            self.use_neural_net = False

    def mutate(self):
        pass

    def reproduce(self, other):
        pass

    def fitness_measure(self):
        pass

    @staticmethod
    def init_random():
        pass
