from keras.preprocessing.text import one_hot


class FeatureGenerator:
    @staticmethod
    def one_hot_encoding(urls: list, vocab_size=87) -> list:
        """Integer encode the documents"""
        encoded_docs = [one_hot(str(d), vocab_size) for d in urls]
        return encoded_docs
