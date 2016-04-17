from organism import Organism
from ga import GA
import numpy as np
from glob import glob

def loadData():
    fileList = glob('../data/datasets/*.csv')
    for f in fileList:
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        X_data = data[:, :-1].astype('float32')
        Y_data = data[:, -1].astype('float32')
        Y_data[Y_data > 0] = 1
        dataDict = {'X': X_data, 'Y': Y_data}
        yield dataDict

def main():
    pop_size, gen_count, mutation = 10, 5, 0.3
    solve_methods = ["keras", "pybrain", "svm", "logistic", "naivebayes", \
                     "randomforest", "lda"]
    for dataset in loadData():
        Organism.data = dataset
        Organism.count = dataset['X'].shape[1]
        for solve_method in solve_methods:
            print "Using solve method: ", solve_method
            full_accuracy = GA.full_accuracy(solve_method)
            print "Accuracy using all features: ", full_accuracy.fitness
            solver = GA(gen_count, pop_size, mutation, solve_method)
            finalPop = solver.search()
            print "Best Accuracy: ", finalPop[0].fitness
            print "Subset of features used: ", finalPop[0].feature_subset

if __name__ == '__main__':
    main()
