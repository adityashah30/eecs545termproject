from classifier import classifier_factory
import numpy as np

class Organism:
    '''
    The Organism class. Encodes the following information.
    1. The subset of the features used for classification.
    2. Optional: If the classifier is a neural network, the number of
                 nodes in the hidden layer.
    '''

    data = None
    count = None  #number of features
    mutation = 0.3

    def __init__(self, feature_subset=[1,2,3,4,5], hidden_nodes=10, solve_method="logistic"):
        self.feature_subset = feature_subset
        self.hidden_nodes = hidden_nodes
        self.solve_method = solve_method
        self.fitness = self.fitness_measure()

    def fitness_measure(self):
        classifier = classifier_factory(self.solve_method, self.hidden_nodes)
        return classifier.fit_score(Organism.data, self.feature_subset)

    @staticmethod
    def init_random(subset_size, solve_method="logistic"):
        hidden_nodes = np.random.randint(subset_size, Organism.count)
        features = np.random.randint(0, Organism.count-1, size=subset_size)
        return Organism(features, hidden_nodes, solve_method)

    @staticmethod
    def mutate(feature_subset):
        if np.random.random() <= Organism.mutation:
            new_feature = np.random.randint(0, Organism.count-1)
            rand_idx = np.random.randint(0, feature_subset.shape[0]-1)
            if new_feature not in feature_subset:
                feature_subset[rand_idx] = new_feature
        return feature_subset

    @staticmethod
    def reproduce(org1, org2):
        size = np.minimum(len(org1.feature_subset), len(org2.feature_subset))
        crossover_point = np.random.randint(1, size-1)
        new_features = np.concatenate((org1.feature_subset[:crossover_point], org2.feature_subset[crossover_point:]))
        new_features = np.unique(new_features)
        new_features = Organism.mutate(new_features)
        new_hidden_nodes = max(org1.hidden_nodes, org2.hidden_nodes)
        return Organism(new_features, new_hidden_nodes, org1.solve_method)
