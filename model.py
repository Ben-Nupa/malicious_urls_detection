import os
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Dropout, Lambda
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras import backend
from keras.callbacks import TensorBoard
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from feature_generator import FeatureGenerator


class UrlDetector:
    def __init__(self, model="simple_nn", vocab_size=87, max_length=200):
        """
        Initiates URL detector model. Default parameters values are taken from J. Saxe et al. - eXpose: A Character-
        Level Convolutional Neural Network with Embeddings For Detecting Malicious URLs, File Paths and Registry Keys

        Parameters
        ----------
        model: {"simple_nn", "big_conv_nn"}
            Path of csv file containing the dataset.
        max_length:
            Maximum length of considered URL (crops longer URL).
        vocab_size:
            Size of alphabet (letters, digits, symbols...).
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.model = Sequential()
        self.build_model(model)

    def build_model(self, model: str):
        """
        Builds given model.

        Parameters
        ----------
        model: {"simple_nn", "big_conv_nn"}
            Path of csv file containing the dataset.
        """
        if model == "simple_nn":
            self._build_simple_nn()
        elif model == "big_conv_nn":
            self._build_big_conv_nn()

    def _build_simple_nn(self):
        """Defines and compiles a simple NN."""
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, 32, input_length=self.max_length))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        print(self.model.summary())

    def _get_complete_conv_layer(self, filter_length, nb_filter):
        """Wrap up for convolutional layer followed with a summing pool layer, batch normalization and dropout."""
        model = Sequential()
        model.add(Convolution1D(nb_filter=nb_filter,
                                input_shape=(self.max_length, 32),
                                filter_length=filter_length,
                                border_mode='same',
                                activation='relu',
                                subsample_length=1))
        model.add(BatchNormalization())
        model.add(Lambda(self._sum_1d, output_shape=(nb_filter,)))
        # model.add(BatchNormalization(mode=0))
        model.add(Dropout(0.5))
        return model

    @staticmethod
    def _sum_1d(x):
        """Sum layers on column axis."""
        return backend.sum(x, axis=1)

    def _build_big_conv_nn(self, optimizer="adam", compile=True):
        """
        Defines and compiles same CNN as J. Saxe et al. - eXpose: A Character-Level Convolutional Neural Network with
        Embeddings For Detecting Malicious URLs, File Paths and Registry Keys.

        Parameters
        ----------
        optimizer:
            Optimizer algorithm
        """
        main_input = Input(shape=(self.max_length,), dtype='int32', name='main_input')
        embedding = Embedding(input_dim=self.vocab_size, output_dim=32, input_length=self.max_length,
                              dropout=0)(main_input)

        conv1 = self._get_complete_conv_layer(2, 256)(embedding)
        conv2 = self._get_complete_conv_layer(3, 256)(embedding)
        conv3 = self._get_complete_conv_layer(4, 256)(embedding)
        conv4 = self._get_complete_conv_layer(5, 256)(embedding)

        merged = merge.Concatenate()([conv1, conv2, conv3, conv4])
        merged = BatchNormalization()(merged)

        middle = Dense(1024, activation='relu')(merged)
        middle = BatchNormalization()(middle)
        middle = Dropout(0.5)(middle)

        middle = Dense(1024, activation='relu')(middle)
        middle = BatchNormalization()(middle)
        middle = Dropout(0.5)(middle)

        middle = Dense(1024, activation='relu')(middle)
        middle = BatchNormalization()(middle)
        middle = Dropout(0.5)(middle)

        output = Dense(1, activation='sigmoid')(middle)

        self.model = Model(input=main_input, output=output)
        if compile:
            self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
        print(self.model.summary())

    def _get_padded_docs(self, encoded_docs: list) -> list:
        """Makes the data readable for the model."""
        padded_docs = pad_sequences(encoded_docs, maxlen=self.max_length, padding='post')
        return padded_docs

    def fit(self, encoded_docs: list, labels: list, epochs=5, verbose=1, training_logs="training_logs"):
        """
        Trains the model with Tensorboard monitoring. Data should be shuffled before calling this function because the
        validation set is taken from the last samples of the provided dataset.

        Parameters
        ----------
        encoded_docs
            One-hot encoded URLs.
        labels
            Labels (0/1) of URLs.
        epochs
            Number of epochs to train on.
        verbose
            Whether to display information (loss, accuracy...) during training.
        training_logs:
            Directory where to store Tensorboard logs.
        """
        if not os.path.exists(training_logs):
            os.makedirs(training_logs)
        tensorboard = TensorBoard(log_dir=training_logs)
        padded_docs = self._get_padded_docs(encoded_docs)
        self.model.fit(padded_docs, labels, epochs=epochs, validation_split=0.2, verbose=verbose,
                       callbacks=[tensorboard])

    def compute_accuracy(self, encoded_docs: list, labels: list):
        """Computes the accuracy of given data."""
        padded_docs = self._get_padded_docs(encoded_docs)
        loss, accuracy = self.model.evaluate(padded_docs, labels, verbose=1)
        print('Accuracy: %f' % (accuracy * 100))
        # TODO : compute F1-score, ROC curve

    def predict_proba(self, encoded_docs: list):
        """Predicts the probabilities of given data."""
        padded_docs = self._get_padded_docs(encoded_docs)
        probabilities = self.model.predict_proba(padded_docs)
        return probabilities

    def plot_roc_curve(self, encoded_docs: list, labels: list):
        """Plots the ROC curve and computes its AUC."""
        probabilities = self.predict_proba(encoded_docs)
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)
        # Figure
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")


if __name__ == '__main__':
    # Data
    feature_generator = FeatureGenerator()
    urls, labels = feature_generator.load_data(os.path.join("datasets", "url_data_mega_deep_learning.csv"),
                                               url_column_name="url",
                                               label_column_name="isMalicious",
                                               to_binarize=False)
    one_hot_urls = feature_generator.one_hot_encoding(urls)

    # Model
    url_detector = UrlDetector("big_conv_nn")
    url_detector.fit(one_hot_urls, labels)
    url_detector.compute_accuracy(one_hot_urls[-100:], labels[-100:])

    plt.show()
