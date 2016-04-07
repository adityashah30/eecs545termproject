This document serves as a guide for the project.

Basically for the project, we need to select the features that result in best classification
There are some features that not only do not contribute anything to the classification model
but deteriorate its performance due to random noise. As a result they should not be included
in the model. As a result the selection of features becomes an important task.

Since our aim is to create a software bug classification model, in order to manually select the
features, we would need prior knowledge of the code. In most software projects this is not possible
for a person to have idea about the whole project. As a result, we need to use automated tools to
assist us in this endeavor. 

We use a neural network to implement the classifier using a subset of the features provided.
Since there are so many features, we select the best features using Genetic Algorithms (GA).
The method used in this project is as follows

1. Initialize a population of subsets of features chosen at random.
2. The genome of an individual encodes the following information
   1. The subset of the features used.
   2. The number of nodes in the hidden layer of the neural network.
   3, Number of epochs to train.
3. Individuals are trained on a subset of the training set with the remaining dataset used for
   validation. This serves as the fitness function to determine the fit individuals that live on
   in the next generation.

This is achieved using the following code structure
1. Organism class encodes all the information about mutation and crossovers and fitness of an 
   individual.
2. Population class encodes all the information about managing the population (collection of 
   Organisms) from one generation to the next.
3. GA class that uses the Population class and implements the actual Genetic Algorithm.
4. FitnessMeasure class that uses a means of classification including but not limited to Neural
   Networks to evaluate the fitness of an individual Organism by training the said model using 
   the parameters encoded in the Organism's genotype and returns a validation accuracy score on
   validation dataset which is a subset of the training dataset.

