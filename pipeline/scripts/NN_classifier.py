#!/usr/bin/env python.

import os
from dotenv import load_dotenv
import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import tensorflow as tf

load_dotenv()
ROOT = os.environ.get("ROOT")
# Seen repos
TEXT_EMBEDDINGS_SEEN = f"{ROOT}/pipeline/pickles/text_embeddings_seen.pkl"
TITLE_EMBEDDINGS_SEEN = f"{ROOT}/pipeline/pickles/title_embeddings_seen.pkl"
WORD_COUNT_VECTORS_SEEN = f"{ROOT}/pipeline/pickles/word_count_vectors_seen.pkl"
# HANDCRAFTED_FEATURES_FILE = f"{ROOT}/pipeline/pickles/handcrafted_features.pkl"
TITLE_SENTENCE_EMBEDDINGS_SEEN= f"{ROOT}/pipeline/pickles/title_sentence_embeddings_seen.pkl"
BODY_SENTENCE_EMBEDDINGS_SEEN= f"{ROOT}/pipeline/pickles/body_sentence_embeddings_seen.pkl"

# Seen repos
TEXT_EMBEDDINGS_UNSEEN = f"{ROOT}/pipeline/pickles/text_embeddings_unseen.pkl"
TITLE_EMBEDDINGS_UNSEEN = f"{ROOT}/pipeline/pickles/title_embeddings_unseen.pkl"
WORD_COUNT_VECTORS_UNSEEN = f"{ROOT}/pipeline/pickles/word_count_vectors_unseen.pkl"
TITLE_SENTENCE_EMBEDDINGS_UNSEEN= f"{ROOT}/pipeline/pickles/title_sentence_embeddings_unseen.pkl"
BODY_SENTENCE_EMBEDDINGS_UNSEEN= f"{ROOT}/pipeline/pickles/body_sentence_embeddings_unseen.pkl"

def load_pickle(filename):
    retrieved_df = pd.read_pickle(filename)

    # Manually converting Feature array to suitable format due to some errors at model.fit()
    raw_X = retrieved_df['Feature']
    processed_X = []
    for row in raw_X:
        feature_vector = []
        for value in row:
            feature_vector.append(float(value))
        processed_X.append(feature_vector)

    return processed_X, retrieved_df['Label']

# pass in an array of filepaths as strings
def combine_pickles(files):
    X_dataframes = []
    df_y = [] # df_y is the same for all files
    for filename in files:
        df_X, df_y = load_pickle(filename)
        X_dataframes.append(df_X)

    new_X = []
    for i in range(len(df_y)):
        feature_vector = []
        for df in X_dataframes:
            feature_vector += df[i]
        new_X.append(feature_vector)

    return new_X, df_y

def main():
    # Load training data. NOTE: only pass in seen repos here
    print("Combining pickles...")
    df_X_seen, df_y_seen = combine_pickles([TEXT_EMBEDDINGS_SEEN, WORD_COUNT_VECTORS_SEEN, TITLE_EMBEDDINGS_SEEN])

    # Train-Test split
    # [NOTE: No need to randomise as randomisation has already been done in scripts/dataframe_generator.py]
    training_length = math.ceil(len(df_X_seen) * 0.8)
    X_train = df_X_seen[:training_length]
    y_train = df_y_seen[:training_length]
    X_test_seen = df_X_seen[training_length:]
    y_test_seen = df_y_seen[training_length:]

    # convert to numpy array to fit into the NN
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test_seen = np.asarray(X_test_seen)
    y_test_seen = np.asarray(y_test_seen)

    # Training
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(798,)),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    y_train = y_train.astype('int')
    print("X_train length: ", len(X_train))
    # print("10 examples of X_train: ", X_train[:10])
    print("X_train feature length: ", len(X_train[0]))
    print("y_train length: ", len(y_train))
    print("Training model now...")
    model.fit(X_train, y_train, epochs=5)

    # Testing for seen repos
    print("Testing model on testing set of SEEN repos now...")
    model.evaluate(X_test_seen, y_test_seen)

    # sanity checks
    X_test_seen = None
    y_test_seen = None

    # Testing for unseen repos
    # load unseen repos first
    df_X_unseen, df_y_unseen = combine_pickles([TEXT_EMBEDDINGS_UNSEEN, WORD_COUNT_VECTORS_UNSEEN, TITLE_EMBEDDINGS_UNSEEN])
    df_X_unseen = np.asarray(df_X_unseen)
    df_y_unseen = np.asarray(df_y_unseen)
    print("length of df_X_unseen: ", len(df_X_unseen)) # sanity check
    print("Testing model on UNSEEN repos now...")
    model.evaluate(df_X_unseen, df_y_unseen)

if __name__ == "__main__":
    main()
