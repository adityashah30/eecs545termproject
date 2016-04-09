Iimport numpy as np

class Organism:
    '''
    The Organism class. Encodes the following information.
    1. The subset of the features used for classification.
    2. Optional: If the classifier is a neural network, the number of
                 nodes in the hidden layer.
    '''

    def __init__(self, count,feature_subset=[], hidden_nodes=None):
        self.feature_subset = feature_subset
        self.count = count #num of features
        if hidden_nodes:
            self.hidden_nodes = hidden_nodes
            self.use_neural_net = True
        else:
            self.use_neural_net = False

    def mutate(self):
        subsetsize = len(feature_subset)
        index = np.random.randint(0,subsetsize)
        feature = np.random.randint(0,self.count)
        if feature in self.feature_subset:
            feature = np.random.randint(0,self.count)
        else
            self.feature_subset[index] = feature

        pass

    def reproduce(self, other):
        subsetsize1 = len(self.feature_subset)
        subsetsize2 = len(other.feature_subset)
        size = np.maximum(subsetsize1,subsetsize2)
        crossoverpoint = np.random.randint(1,size-1)
        childfeatures = []
        childfeatures = self.feature_subset[:crossoverpoint] + other.feature_subset[crossoverpoint:]
        childcount = len(childfeatures)
        child = Organism(childcount,childfeatures)
        return child
        #pass

    def fitness_measure(self):
        pass

    @staticmethod
    def init_random(subsetsize):
        count = 40 #num of features
        features = np.random.randint(0,count,size = subsetsize)
        Org = Organism(count,features)




        pass
