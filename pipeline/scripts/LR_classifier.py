#!/usr/bin/env python.

import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

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

def combine_pickles(file1, file2, file3):
    # df_y is the same 
    df_X1, df_y = load_pickle(file1)
    df_X2, df_y = load_pickle(file2)
    df_X3, df_y = load_pickle(file3)
    
    new_X = []
    for i in range(len(df_y)):
        new_X.append(df_X1[i] + df_X2[i] + df_X3[i])
    
    return new_X, df_y

def main():
    # Load data
    # df_X, df_y = load_pickle(filename="../pickles/text_embeddings.pkl")
    df_X, df_y = combine_pickles(file1="../pickles/text_embeddings.pkl", 
        file2="../pickles/title_embeddings.pkl", 
        file3="../pickles/word_count_vectors.pkl")
    
    #### analysis #####
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
    print('accurracy score on validation = {}'.format(score))

if __name__ == "__main__":
    main()
