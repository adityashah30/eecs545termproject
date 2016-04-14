from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
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

    def __init__(self, hidden_net=31):
        self.hidden_net = hidden_net
        self.seed = 42
        self.nb_classes = 2
        self.batch_size = 64
        self.nb_epoch = 10

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
        Y_test = np_utils.to_categorical(Y_test, self.nb_classes)

        nb_features = X_train.shape[1]

        model = Sequential()
        model.add(Dense(self.hidden_net, input_shape=(nb_features,)))
        model.add(Dropout(0.15))
        model.add(Activation('relu'))
            
        model.add(Dense(self.hidden_net))
        model.add(Dropout(0.15))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(class_mode='binary', loss='binary_crossentropy', optimizer='rmsprop')

        model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch, show_accuracy=True, verbose=0, validation_data=(X_val, Y_val))

        # test_score = model.evaluate(X_test, Y_test, verbose=0, batch_size=self.batch_size)
        Y_pred = model.predict_classes(X_test, batch_size=self.batch_size, verbose=0)
        test_score = roc_auc_score(Y_test, Y_pred)
        print test_score
        return test_score
