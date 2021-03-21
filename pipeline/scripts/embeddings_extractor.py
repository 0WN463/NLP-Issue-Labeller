#!/usr/bin/env python.

import os
import math
import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

###### This script generates /pickles/text_embeddings.pkl AND /pickles/title_embeddings.pkl######

LABELS = {
    "feature": 0,
    "bug": 1,
    "doc": 2
}

# Given an array of sentences, transform each sentence into a vector
# representing the average of normalised embeddings of all the words.
def generate_averaged_word_embeddings(embeddings_model, sentences):
    def normalise(x):
        sum_of_squares = np.sum(x**2)
        l2_norm = np.sqrt(sum_of_squares)
        if l2_norm > 0:
            return (1.0 / l2_norm) * x
        else:
            return x

    ###### Discarded: reweighting using idf values ########
    # train = pd.read_csv('./data/train.csv')
    # X_train = train['Text']
    # tf = TfidfVectorizer(use_idf=True)
    # tf.fit_transform(X_train)
    # idf = tf.idf_
    
    res = []
    num_of_sentences = 0
    for sentence in sentences:
        num_of_sentences += 1 

        pattern = r"\w+'\w+|\w+-\w+|\w+|[(...).,!:\";\?]"
        tokens = re.findall(pattern, sentence)
        summed_vector = None
        is_first = True
        length = 0

        for token in tokens:
            if token in embeddings_model:
                embedding = normalise(embeddings_model[token])
                ###### Discarded: reweighting using idf values (cont. from above) ########
                # idf_value = 1
                # if token.lower() in tf.vocabulary_:
                #     idf_value = idf[tf.vocabulary_[token.lower()]]
                # embedding = embedding * idf_value
                length += 1
                if is_first:
                    summed_vector = embedding
                    is_first = False
                else:
                    summed_vector = summed_vector + embedding
        if summed_vector is None:
            summed_vector = np.zeros(300)
            length = 1
        averaged_vector = summed_vector / length
        res.append(averaged_vector)

        if num_of_sentences % 1000 == 0:
            print("Done with vectorising # of sentences: ", num_of_sentences)

    return res

# Given a String of sentence, returns an array of tokens
def tokenise(sentence):
    pattern = r'\w+'
    return re.findall(pattern, sentence)

# sg=1 for skip-gram; sg=0 for CBOW
def train_embeddings(df, size, window, sg):
    sentences = []
    for _, row in df.iterrows():
        sentences.append(tokenise(row['text']))
        sentences.append(tokenise(row['title']))

    embeddings_model = Word2Vec(sentences=sentences, size=size, window=window, sg=sg)
    return embeddings_model

def load_dataframe_from_pickle():
    retrieved_df = pd.read_pickle("../pickles/dataframe.pkl")
    return retrieved_df

def save_vector_array(vector_array, labels, filename):
    save_df = pd.DataFrame(columns=['Feature', 'Label'])
    save_df['Feature'] = pd.Series(vector_array)
    save_df['Label'] = labels.values
    save_df.to_pickle(filename)

def main():
    df = load_dataframe_from_pickle()
    print("Done loading dataframe.")

    embeddings_model = train_embeddings(df, size=300, window=5, sg=0) # 0 for Skip-gram
    print("Done with embeddings training.")

    text_vector_array = generate_averaged_word_embeddings(embeddings_model, df['text'])
    print("Done with text embedding vectorisation.")
    title_vector_array = generate_averaged_word_embeddings(embeddings_model, df['title'])
    print("Done with title embedding vectorisation.")

    print("Saving text vector array to memory...")
    save_vector_array(text_vector_array, df['labels'], filename="../pickles/text_embeddings.pkl")
    print("Done with saving")

    print("Saving title vector array to memory...")
    save_vector_array(title_vector_array, df['labels'], filename="../pickles/title_embeddings.pkl")
    print("Done with saving")

if __name__ == "__main__":
    main()
