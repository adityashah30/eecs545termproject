from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop

class Classifier:
    '''
    The Classifier class. Takes in a genotype and trains a classifier model
    on a subset of the training set and returns evaluation score on test set
    which is again derived from the training set.
    '''

    def __init__(self, hidden_net=[100]):
        self.hidden_net = hidden_net
        self.seed = 42
        self.nb_classes = 2
        self.batch_size = 64
        self.nb_epoch = 10

    def fit_score(self, data, feature_set):
        X, Y = data['X'][feature_set], data['Y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.125, random_state=self.seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        Y_train = np_utils.to_categorical(Y_train, self.nb_classes)
        Y_val = np_utils.to_categorical(Y_val, self.nb_classes)
        Y_test = np_utils.to_categorical(Y_test, self.nb_classes)

        nb_features = X_train.shape[0]

        model = Sequential()
        model.add(Dense(hidden_net[0], input_shape=(nb_features,)))
        model.add(Dropout(0.15))
        model.add(Activation('relu'))

        for nb_nodes in self.hidden_net:
            model.add(Dense(nb_nodes))
            model.add(Dropout(0.15))
            model.add(Activation('relu'))

        model.add(Dense(self.nb_classes))
        model.add(Activation('signmoid'))

        model.compile(loss='binary_crossentropy', optimizer='rmsprop')

        model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_val, Y_val))

        return model.evaluate(X_test, Y_test, batch_size=self.batch_size)





