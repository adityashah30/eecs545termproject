from organism import Organism
from ga import GA
import numpy as np
from glob import glob

def loadData():
    fileList = glob('../data/datasets/*.csv')
    for f in fileList:
        print "Loading file: ", f
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        X_data = data[:, :-1].astype('float32')
        Y_data = data[:, -1].astype('float32')
        Y_data[Y_data > 0] = 1
        dataDict = {'X': X_data, 'Y': Y_data, 'fname': f}
        yield dataDict

def main():
    pop_size, gen_count, mutation = 1, 1, 0.3
    solve_methods = ["keras", "pybrain", "svm", "logistic", "naivebayes", \
                     "randomforest", "lda"]
    accuracy_fname = "accuracy.txt"
    with open(accuracy_fname, "w") as fp:
        for dataset in loadData():
            fp.write("===================================\n")
            fp.write("Filename: %s\n" % dataset['fname'])
            Organism.data = dataset
            Organism.count = dataset['X'].shape[1]
            fp.write("Num Features: %d\n" % Organism.count)
            fp.write("\n------------------------------------\n")
            for solve_method in solve_methods:
                print "Using solve method: ", solve_method
                fp.write("Using solve method: %s\n" % solve_method)
                full_accuracy = GA.full_accuracy(solve_method)
                print "Accuracy using all features: ", full_accuracy.fitness
                fp.write("Accuracy using all features: %f\n" % full_accuracy.fitness)
                solver = GA(gen_count, pop_size, mutation, solve_method)
                finalPop = solver.search()
                print "Best Accuracy: ", finalPop[0].fitness
                print "Subset of features used: ", finalPop[0].feature_subset
                fp.write("Pop Size: %d; Generation Count: %d; Mutation Rate: %f\n" % (pop_size, gen_count, mutation))
                fp.write("Best Accuracy: %f\n" % finalPop[0].fitness)
                fp.write("Subset of features used: " + str(finalPop[0].feature_subset))
                fp.write("\n------------------------------------\n\n")

if __name__ == '__main__':
    main()
