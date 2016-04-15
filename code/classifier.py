from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD
from sklearn import svm
from sklearn.linear_model import LogisticRegression

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
        self.batch_size = 64
        self.nb_epoch = 20

    def fit_score(self, data, feature_set):
        X, Y = data['X'][:,feature_set], data['Y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.125, random_state=self.seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
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

        # model.add(Dense(self.nb_classes))
        # model.add(Activation('softmax'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')

        model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_val, Y_val))

        # test_score = model.evaluate(X_test, Y_test_cat, verbose=0, batch_size=self.batch_size)
        Y_pred = model.predict_classes(X_test, batch_size=self.batch_size, verbose=1)
        test_score = roc_auc_score(Y_test, Y_pred)
        # print test_score
        return test_score

class NNPybrain:
    '''
    The NNPybrain class. Takes in a genotype and trains a classifier model
    on a subset of the training set and returns evaluation score on test set
    which is again derived from the training set.
    '''

    def __init__(self, hidden_net=31):
        self.hidden_net = hidden_net
        self.seed = 42
        self.nb_classes = 2
        self.batch_size = 64
        self.nb_epoch = 20

    def fit_score(self, data, feature_set):
        X, Y = data['X'][:,feature_set], data['Y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.125, random_state=self.seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
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

        # model.add(Dense(self.nb_classes))
        # model.add(Activation('softmax'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')

        model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_val, Y_val))

        # test_score = model.evaluate(X_test, Y_test_cat, verbose=0, batch_size=self.batch_size)
        Y_pred = model.predict_classes(X_test, batch_size=self.batch_size, verbose=1)
        test_score = roc_auc_score(Y_test, Y_pred)
        # print test_score
        return test_score

class SVMClassifier:
    '''
    The SVMClassifier class. Takes in a genotype and trains a classifier model
    on a subset of the training set and returns evaluation score on test set
    which is again derived from the training set.
    '''

    def __init__(self):
        self.seed = 42

    def fit_score(self, data, feature_set):
        X, Y = data['X'][:,feature_set], data['Y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        nb_features = X_train.shape[1]

        model = svm.SVC()
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)
        test_score = roc_auc_score(Y_test, Y_pred)
        # print test_score
        return test_score

class LogisitcRegClassifier:
    '''
    The LogisitcRegClassifier class. Takes in a genotype and trains a classifier model
    on a subset of the training set and returns evaluation score on test set
    which is again derived from the training set.
    '''

    def __init__(self):
        self.seed = 42

    def fit_score(self, data, feature_set):
        X, Y = data['X'][:,feature_set], data['Y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        nb_features = X_train.shape[1]

        model = LogisticRegression(n_jobs=7)
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)
        test_score = roc_auc_score(Y_test, Y_pred)
        # print test_score
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
