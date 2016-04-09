from classifier import Classifier
import numpy as np
from glob import glob

def loadData():
    fileList = glob('../data/nasa/pc5.csv')
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
        classifier = Classifier(self.hidden_nodes)
        #0:loss, 1:accuracy
        return classifier.fit_score(Organism.data, self.feature_subset)[1]

    @staticmethod
    def init_random():
        pass    
