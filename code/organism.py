from classifier import Classifier
import numpy as np
from glob import glob

def loadData():
    fileList = glob('../data/use_greedy_1/*.csv')
    data = np.concatenate([np.loadtxt(f, dtype='str', delimiter=',', skiprows=1) for f in fileList])
    X_data = data[:, :-2].astype('float32')
    Y_data = data[:, -1].astype('int8')
    dataDict = {'X': X_data, 'Y': Y_data}
    return dataDict

class Organism:
    '''
    The Organism class. Encodes the following information.
    1. The subset of the features used for classification.
    2. Optional: If the classifier is a neural network, the number of
                 nodes in the hidden layer.
    '''

    data = loadData()

    def __init__(self, count,feature_subset=[], hidden_nodes=None):
    	data = loadData()
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
        else:
            self.feature_subset[index] = feature

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

    def fitness_measure(self):
        classifier = Classifier(self.hidden_nodes)
        #0:loss, 1:accuracy
        return classifier.fit_score(Organism.data, self.feature_subset)[1]

    @staticmethod
    def init_random(subsetsize):
        count = 40 #num of features
        features = np.random.randint(0,count,size = subsetsize)
        return Organism(count,features)

