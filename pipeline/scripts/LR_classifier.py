#!/usr/bin/env python.

import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

load_dotenv()
ROOT = os.environ.get("ROOT")
TEXT_EMBEDDINGS_FILE = f"{ROOT}/pipeline/pickles/text_embeddings.pkl"
TITLE_EMBEDDINGS_FILE = f"{ROOT}/pipeline/pickles/title_embeddings.pkl"
WORD_COUNT_VECTORS_FILE = f"{ROOT}/pipeline/pickles/word_count_vectors.pkl"
HANDCRAFTED_FEATURES_FILE = f"{ROOT}/pipeline/pickles/handcrafted_features.pkl"
TITLE_SENTENCE_EMBEDDINGS_FILE= f"{ROOT}/pipeline/pickles/title_sentence_embeddings.pkl"
TEXT_SENTENCE_EMBEDDINGS_FILE= f"{ROOT}/pipeline/pickles/text_sentence_embeddings.pkl"

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
    # Load data
    # df_X, df_y = load_pickle(filename="../pickles/handcrafted_features.pkl")
    print("Combining pickles...")
    df_X, df_y = combine_pickles([TEXT_EMBEDDINGS_FILE, TITLE_EMBEDDINGS_FILE, WORD_COUNT_VECTORS_FILE, TITLE_SENTENCE_EMBEDDINGS_FILE, TEXT_SENTENCE_EMBEDDINGS_FILE])

    #### Analysis #####
    # print(pd.Series(df_y).value_counts())
    # RESULT:
    # 1    33842 (bug)
    # 0    16926 (feature)
    # 2     3382 (doc)

    # Train-Test split
    # [NOTE: No need to randomise as randomisation has already been done in scripts/dataframe_generator.py]
    training_length = math.ceil(len(df_X) * 0.8)
    X_train = df_X[:training_length]
    y_train = df_y[:training_length]
    X_test = df_X[training_length:]
    y_test = df_y[training_length:]

    # Training
    model = LogisticRegression(C=1.3, max_iter=2000)
    # model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        # max_depth=1, random_state=0)
    print("Training model now...")
    print("X_train length: ", len(X_train))
    # print("10 examples of X_train: ", X_train[:10])
    print("X_train feature length: ", len(X_train[0]))
    print("y_train length: ", len(y_train))
    y_train = y_train.astype('int')
    model.fit(X_train, y_train)

    # Testing
    print("Testing model now...")
    y_pred = model.predict(X_test)

    # Use accurracy as the metric
    y_test = y_test.astype('int')
    score = accuracy_score(y_test, y_pred)
    print('Accurracy score on test set = {}'.format(score))


if __name__ == "__main__":
    main()
