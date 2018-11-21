import pandas as pd
from keras.preprocessing.text import one_hot


class FeatureGenerator:
    @staticmethod
    def load_data(dataset: str, url_column_name="url", label_column_name="label", to_binarize=False,
                  neg_word="bad") -> tuple:
        """
        Load given data file into self.urls and self.labels

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

        dataframe: pd.DataFrame = pd.read_csv(dataset)
        urls = dataframe[url_column_name].tolist()
        labels = list(map(binarize_list, dataframe[label_column_name].tolist())) if to_binarize else dataframe[
            label_column_name].tolist()
        return urls, labels

    @staticmethod
    def one_hot_encoding(urls: list, vocab_size=87) -> list:
        """Integer encode the documents"""
        encoded_docs = [one_hot(str(d), vocab_size) for d in urls]
        return encoded_docs
