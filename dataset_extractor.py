import pandas as pd
from datetime import datetime, date
from sklearn.utils import shuffle


def load_data(dataset: str, url_column_name="url", label_column_name="label", to_binarize=False, neg_word="bad",
              to_shuffle=True) -> tuple:
    """
    Load given data file.

    Parameters
    ----------
    dataset
        Path of csv file containing the dataset.
    url_column_name
        Name of the column containing the urls.
    label_column_name
        Name of the column containing the labels.
    to_binarize
        True if the label column is not already in binary form.
    neg_word
        Negative word in the label column. Only considered if 'to_binarize' is True.
    to_shuffle
        Whether to shuffle the dataset.

    Returns
    -------
    tuple
        (list containing the urls, list containing the labels)
    """

    def binarize_list(element: str) -> int:
        """Binarize given element."""
        if element == neg_word:
            return 1
        else:
            return 0

    dataframe = pd.read_csv(dataset)
    urls = dataframe[url_column_name].tolist()
    labels = list(map(binarize_list, dataframe[label_column_name].tolist())) if to_binarize else \
        dataframe[label_column_name].tolist()
    if to_shuffle:
        urls, labels = shuffle(urls, labels)
    return urls, labels


def load_dated_data(bad_dataset: str, good_dataset: str, ratio_good_bad: float, separation_date: date) -> tuple:
    """
    Load given dated data file.

    Parameters
    ----------
    bad_dataset
        Path of csv file containing the dataset with bad URLs.
    good_dataset
        Path of csv file containing the dataset with benign URLs.
    ratio_good_bad
        Ratio of (Good Data)/(Bad Data).
    separation_date
        Date for the separation of the training and testing set (before that date, URLs are put in the training set,
        after that date, they are put int the testing set)

    Returns
    -------
    tuple
        (list of training urls, list of training labels, list of testing urls, list of testing labels)
    """
    training_urls = []
    training_labels = []
    testing_urls = []
    testing_labels = []

    # Extracting bad URLS
    bad_dataframe = pd.read_csv(bad_dataset, ";")
    bad_urls = bad_dataframe["URL"].tolist()
    bad_dates = bad_dataframe["Date"].tolist()

    for i in range(len(bad_urls)):
        if datetime.strptime(str(bad_dates[i]), '%Y%m%d').date() < separation_date:
            # Malicious URLs before a certain date go into the training set
            training_urls.append(bad_urls[i])
            training_labels.append(0)
        else:
            # Malicious URLs after a certain date go into the testing set
            testing_urls.append(bad_urls[i])
            testing_labels.append(0)

    # Extracting good URLS
    good_dataframe = pd.read_csv(good_dataset, ";")
    good_urls = good_dataframe["URL"].sample(int(len(bad_urls) * ratio_good_bad)).tolist()
    idx_seperation = int(len(training_urls) * ratio_good_bad)

    training_urls += good_urls[:idx_seperation]
    training_labels += [1 for i in range(len(good_urls[:idx_seperation]))]

    testing_urls += good_urls[idx_seperation:]
    testing_labels += [1 for i in range(len(good_urls[idx_seperation:]))]

    # Shuffle
    training_urls, training_labels = shuffle(training_urls, training_labels)
    testing_urls, testing_labels = shuffle(testing_urls, testing_labels)

    print("Created a dataset with separation date " + str(separation_date) + " and testing set represents " +
          str(100 * len(testing_urls) / (len(bad_urls) + len(good_urls))) + "% of original dataset.")
    return training_urls, training_labels, testing_urls, testing_labels
