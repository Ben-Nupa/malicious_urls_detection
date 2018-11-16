import pandas as pd
from keras.preprocessing.text import one_hot
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class FeatureGenerator:
    def load_data(self, dataset: str, url_column_name="url", label_column_name="label", to_binarize=False) -> tuple:
        """Load given data file into self.urls and self.labels"""

        def binarize_list(element: str) -> int:
            if element == "bad":
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

    @staticmethod
    def build_lexical_features(data):
        vectorizer = CountVectorizer()
        vectorizer.fit_transform(data)
