from classifier import Classifier
import numpy as np
from glob import glob


def loadData():
    fileList = glob('eclipseJDT.csv')
    data = np.concatenate([np.loadtxt('eclipseJDT.csv', delimiter=',', skiprows=1) for f in fileList])
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
    count = 32  #number of features

    def __init__(self, feature_subset=[], hidden_nodes=None):
        data = loadData()
        self.feature_subset = feature_subset
        if hidden_nodes:
            self.hidden_nodes = hidden_nodes
            self.use_neural_net = True
        else:
            self.use_neural_net = False

    def mutate(self):
        subsetsize = len(self.feature_subset)
        index = np.random.randint(0,subsetsize)
        feature = np.random.randint(0,Organism.count)
        while(feature in self.feature_subset):
            feature = np.random.randint(0,Organism.count)
        self.feature_subset[index] = feature

    def reproduce(self, other):
        subsetsize1 = len(self.feature_subset)
        subsetsize2 = len(other.feature_subset)
        size = np.maximum(subsetsize1,subsetsize2)
        crossoverpoint = np.random.randint(1,size-1)
        childfeatures = np.concatenate((self.feature_subset[:crossoverpoint],other.feature_subset[crossoverpoint:]))
        hiddennodes = np.random.randint(1,size-1)
        child = Organism(childfeatures,hiddennodes)
        return child

    def fitness_measure(self):
        classifier = Classifier(self.hidden_nodes)
        #0:loss, 1:accuracy
        return classifier.fit_score(Organism.data, self.feature_subset)[1]

    @staticmethod
    def init_random(subsetsize):
        hiddennodes = np.random.randint(1,subsetsize-1)
        features = np.random.randint(0,Organism.count,size=subsetsize)
        return Organism(features,hiddennodes)

