import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from feature_generator import FeatureGenerator
from dataset_extractor import *
from model import UrlDetector

import numpy as np


def predict_on_non_dated_dataset(non_dated_dataset='url_data_mega_deep_learning.csv'):
    """Prediction on the non-dated dataset."""
    if non_dated_dataset == 'url_data_mega_deep_learning.csv':
        to_binarize = False
        label_column_name = 'isMalicious'
        url_column_name = 'url'
    elif non_dated_dataset == 'simple.csv':
        to_binarize = True
        label_column_name = 'label'
        url_column_name = 'url'

    # Data
    urls, labels = load_data(os.path.join("datasets", non_dated_dataset),
                             url_column_name=url_column_name,
                             label_column_name=label_column_name,
                             to_binarize=to_binarize)

    # Features
    feature_generator = FeatureGenerator()
    one_hot_urls = feature_generator.one_hot_encoding(urls)
    X_train, X_test, y_train, y_test = train_test_split(one_hot_urls, labels, test_size=0.2)

    # Model
    url_detector = UrlDetector("big_conv_nn")
    url_detector.fit(X_train, y_train, epochs=5, batch_size=128)

    # Evaluate
    url_detector.evaluate(X_test, y_test)
    url_detector.plot_roc_curve(X_test, y_test)

    plt.show()


def predict_on_dated_dataset(day=15, month=7, year=2018, randomise=False, ratio_good_bad=1, ratio_testing_set=0.2):
    """
    Prediction on the dated dataset.

    Parameters
    ----------
    day, month, year
        Limit date: the network is trained on data dated before this day and tested on data from this day and newer.
        The proportion training/testing is at 80%/20% for the date 15/07/2018.
        Only used if 'randomise' = False.
    randomise
        Whether to randomise the set so as to use or not the date of the data.
    ratio_good_bad
        Ratio of (Good Data)/(Bad Data).
    ratio_testing_set
        Represents the proportion of the dataset to include in the test split. Only used if 'randomise' = True
    """
    # Data
    if randomise:
        training_urls, training_labels, testing_urls, testing_labels = load_randomized_dated_data(
            os.path.join("datasets", "bad_urls1.csv"), os.path.join("datasets", "good_urls.csv"),
            ratio_good_bad=ratio_good_bad, ratio_testing_set=ratio_testing_set)
    else:
        training_urls, training_labels, testing_urls, testing_labels = load_dated_data(
            os.path.join("datasets", "bad_urls1.csv"), os.path.join("datasets", "good_urls.csv"),
            ratio_good_bad=ratio_good_bad, separation_date=date(year, month, day))  # 20% is at 15/07/2018

    # Features
    feature_generator = FeatureGenerator()
    one_hot_training_urls = feature_generator.one_hot_encoding(training_urls)
    one_hot_testing_urls = feature_generator.one_hot_encoding(testing_urls)

    # Model
    url_detector = UrlDetector("big_conv_nn")
    url_detector.fit(one_hot_training_urls, training_labels, epochs=5, batch_size=128)

    # Evaluate
    url_detector.evaluate(one_hot_testing_urls, testing_labels)
    url_detector.plot_roc_curve(one_hot_testing_urls, testing_labels)

    plt.show()


def cross_datasets(direction='dated-->non_dated', day=15, month=7, year=2018, randomise=False,
                   non_dated_dataset='url_data_mega_deep_learning.csv'):
    """
    Trains the model on a dataset and predicts on the other. Check above functions for the details of the parameters.
    """
    # Parameters definition for non-dated dataset
    if non_dated_dataset == 'url_data_mega_deep_learning.csv':
        to_binarize = False
        label_column_name = 'isMalicious'
        url_column_name = 'url'
    elif non_dated_dataset == 'simple.csv':
        to_binarize = True
        label_column_name = 'label'
        url_column_name = 'url'

    if direction == 'non_dated-->dated':
        # TRAINING
        # Data
        urls, labels = load_data(os.path.join("datasets", non_dated_dataset),
                                 url_column_name=url_column_name,
                                 label_column_name=label_column_name,
                                 to_binarize=to_binarize)

        # Features
        feature_generator = FeatureGenerator()
        one_hot_urls = feature_generator.one_hot_encoding(urls)
        X_train, X_val, y_train, y_val = train_test_split(one_hot_urls, labels, test_size=0.2)

        # Model
        url_detector = UrlDetector("big_conv_nn")
        url_detector.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_val, y_val))

        # TESTING
        # Data
        if randomise:
            urls_1, labels_1, urls_2, labels_2 = load_randomized_dated_data(
                os.path.join("datasets", "bad_urls1.csv"), os.path.join("datasets", "good_urls.csv"),
                ratio_good_bad=1, ratio_testing_set=0.2)
        else:
            # For 80%/20% : take 15/07/2018
            # For 50%/50% : take 01/03/2018
            urls_1, labels_1, urls_2, labels_2 = load_dated_data(
                os.path.join("datasets", "bad_urls1.csv"), os.path.join("datasets", "good_urls.csv"),
                ratio_good_bad=1, separation_date=date(year, month, day))
        # Features
        feature_generator = FeatureGenerator()
        one_hot_urls_2 = feature_generator.one_hot_encoding(urls_2)

        # Evaluate
        url_detector.evaluate(one_hot_urls_2, labels_2)
        url_detector.plot_roc_curve(one_hot_urls_2, labels_2)

        plt.show()

    elif direction == 'dated-->non_dated':
        # TRAINING
        # Data
        if randomise:
            training_urls, training_labels, val_urls, val_labels = load_randomized_dated_data(
                os.path.join("datasets", "bad_urls1.csv"), os.path.join("datasets", "good_urls.csv"),
                ratio_good_bad=1, ratio_testing_set=0.2)
        else:
            training_urls, training_labels, val_urls, val_labels = load_dated_data(
                os.path.join("datasets", "bad_urls1.csv"), os.path.join("datasets", "good_urls.csv"),
                ratio_good_bad=1, separation_date=date(year, month, day))  # 20% is at 15/07/2018

        # Features
        feature_generator = FeatureGenerator()
        one_hot_training_urls = feature_generator.one_hot_encoding(training_urls)
        one_hot_val_urls = feature_generator.one_hot_encoding(val_urls)

        # Model
        url_detector = UrlDetector("big_conv_nn")
        url_detector.fit(one_hot_training_urls, training_labels, epochs=5, batch_size=128,
                         validation_data=(one_hot_val_urls, val_labels))

        # TESTING
        # Data
        urls, labels = load_data(os.path.join("datasets", non_dated_dataset),
                                 url_column_name=url_column_name,
                                 label_column_name=label_column_name,
                                 to_binarize=to_binarize)
        # Features
        size_testing = int(0.21*len(urls))
        feature_generator = FeatureGenerator()
        one_hot_urls = feature_generator.one_hot_encoding(urls)

        # Evaluate
        url_detector.evaluate(one_hot_urls[:size_testing], labels[:size_testing])
        url_detector.plot_roc_curve(one_hot_urls[:size_testing], labels[:size_testing])

        plt.show()


if __name__ == '__main__':
    # SINGLE DATASET
    # To predict on the non-dated dataset
    # predict_on_non_dated_dataset('simple.csv')

    # To predict on the dated dataset, using the date in normal order (train on older data, predict on newer).
    # predict_on_dated_dataset()

    # To predict on the dated dataset but shuffling the data.
    # predict_on_dated_dataset(randomise=True)

    # CROSSING DATASETS
    # Train on non-dated dataset and predict on dated dataset
    # cross_datasets(direction='non_dated-->dated', day=15, month=7, year=2018, non_dated_dataset='simple.csv')

    # Train on non-dated dataset and predict on a random part of the dated but shuffled dataset
    # cross_datasets(direction='non_dated-->dated', randomise=True, non_dated_dataset='simple.csv')

    # Train on dated dataset (older part) and predict on non-dated dataset
    # cross_datasets(direction='dated-->non_dated', day=15, month=7, year=2018)

    # Train on shuffled dated dataset and predict on non-dated dataset
    cross_datasets(direction='dated-->non_dated', randomise=True, non_dated_dataset='simple.csv')
