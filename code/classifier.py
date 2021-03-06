import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import cross_val_score
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA

class NNKeras:
    '''
    The NNKeras class. Takes in a genotype and trains a classifier model
    on a subset of the training set and returns evaluation score on test set
    which is again derived from the training set.
    '''

    def __init__(self, hidden_net=31):
        self.hidden_net = hidden_net
        self.seed = 42
        self.nb_classes = 2
        self.batch_size = 32
        self.nb_epoch = 100

    def fit_score(self, data, feature_set):
        X, Y = data['X'][:,feature_set], data['Y']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.125, random_state=self.seed)

        
        Y_train = np_utils.to_categorical(Y_train, self.nb_classes)
        Y_val = np_utils.to_categorical(Y_val, self.nb_classes)
        Y_test_cat = np_utils.to_categorical(Y_test, self.nb_classes)

        nb_features = X_train.shape[1]

        model = Sequential()
        model.add(Dense(self.hidden_net, input_shape=(nb_features,)))
        # model.add(Dropout(0.2))
        model.add(Activation('sigmoid'))
            
        model.add(Dense(self.hidden_net))
        # model.add(Dropout(0.2))
        model.add(Activation('sigmoid'))

        #model.add(Dense(1))
        #model.add(Activation('sigmoid'))

        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='binary_crossentropy', optimizer='adadelta')

        model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=0, validation_data=(X_val, Y_val))

        test_score = model.evaluate(X_test, Y_test_cat, verbose=0, batch_size=self.batch_size)
        return test_score

class SVMClassifier:
    '''
    The SVMClassifier class. Takes in a genotype and trains a classifier model
    on a subset of the training set and returns evaluation score on test set
    which is again derived from the training set.
    '''

    def __init__(self):
        pass

    def fit_score(self, data, feature_set):
        X, Y = data['X'][:,feature_set], data['Y']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = svm.SVC()

        test_score = cross_val_score(model, X, Y, scoring='roc_auc', cv=10, verbose=0)
        test_score = np.mean(test_score)

        return test_score

class LogisitcRegClassifier:
    '''
    The LogisitcRegClassifier class. Takes in a genotype and trains a classifier model
    on a subset of the training set and returns evaluation score on test set
    which is again derived from the training set.
    '''

    def __init__(self):
        pass

    def fit_score(self, data, feature_set):
        X, Y = data['X'][:,feature_set], data['Y']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = LogisticRegression(n_jobs=-1)

        test_score = cross_val_score(model, X, Y, scoring='roc_auc', cv=10, n_jobs=-1, verbose=0)
        test_score = np.mean(test_score)

        return test_score

class GaussianNBClassifier:
    '''
    The NaiveBayes class. Takes in a genotype and trains a classifier model
    on a subset of the training set and returns evaluation score on test set
    which is again derived from the training set.
    '''

    def __init__(self):
        pass

    def fit_score(self, data, feature_set):
        X, Y = data['X'][:,feature_set], data['Y']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = GaussianNB()

        test_score = cross_val_score(model, X, Y, scoring='roc_auc', cv=10, verbose=0)
        test_score = np.mean(test_score)
        
        return test_score


class RFClassifier:
    '''
    The RnadomForest Classifier class. Takes in a genotype and trains a classifier model
    on a subset of the training set and returns evaluation score on test set
    which is again derived from the training set.
    '''

    def __init__(self):
        pass

    def fit_score(self, data, feature_set):
        X, Y = data['X'][:,feature_set], data['Y']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = RandomForestClassifier()

        test_score = cross_val_score(model, X, Y, scoring='roc_auc', cv=10, verbose=0, n_jobs=-1)
        test_score = np.mean(test_score)

        return test_score

class LDAClassifier:
    '''
    The NaiveBayes class. Takes in a genotype and trains a classifier model
    on a subset of the training set and returns evaluation score on test set
    which is again derived from the training set.
    '''

    def __init__(self):
        pass

    def fit_score(self, data, feature_set):
        X, Y = data['X'][:,feature_set], data['Y']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = LDA()

        test_score = cross_val_score(model, X, Y, scoring='roc_auc', cv=10, verbose=0, n_jobs=-1)
        test_score = np.mean(test_score)

        return test_score


def classifier_factory(solve_method="keras", hidden_nodes=10):
    solve_method = solve_method.lower()
    if solve_method == "keras":
        return NNKeras(hidden_nodes)
    elif solve_method == "pybrain":
        return NNPybrain(hidden_nodes)
    elif solve_method == "svm":
        return SVMClassifier()
    elif solve_method == "logistic":
        return LogisitcRegClassifier()
    elif solve_method == "naivebayes":
        return GaussianNBClassifier()
    elif solve_method == "randomforest":
        return RFClassifier()
    elif solve_method == "lda":
        return LDAClassifier()
    else:
        return LogisitcRegClassifier()
    
