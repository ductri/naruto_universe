from sklearn.feature_extraction.text import CountVectorizer

from naruto_skills.text_transformer import TextTransformer


def build_vocab(preprocessed_docs, file_name, min_df):
    """
    This function assumes each doc in preprocessed_docs can be tokenized by str.split
    :param preprocessed_docs:
    :param file_name:
    :param min_df:
    :return:
    """
    vectorizer = CountVectorizer(min_df=min_df, tokenizer=str.split)
    vectorizer.fit(preprocessed_docs)

    vocabs = list(vectorizer.vocabulary_)
    vocabs.sort()

    text_transformer = TextTransformer(vocabs)
    text_transformer.save_vocab(file_name)
