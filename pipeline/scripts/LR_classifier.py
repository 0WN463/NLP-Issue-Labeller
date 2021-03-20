#!/usr/bin/env python.

import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

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

def main():
    # Load data
    df_X, df_y = load_pickle(filename="../pickles/word_count_vectors.pkl")
    
    # Train-Test split 
    # [NOTE: No need to randomise as randomisation has already been done in scripts/dataframe_generator.py]
    training_length = math.ceil(len(df_X) * 0.8)
    X_train = df_X[:training_length]
    y_train = df_y[:training_length]
    X_test = df_X[training_length:]
    y_test = df_y[training_length:]
    
    # Training
    model = LogisticRegression(C=1.3, max_iter=500)
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
    print('accurracy score on validation = {}'.format(score))

if __name__ == "__main__":
    main()
