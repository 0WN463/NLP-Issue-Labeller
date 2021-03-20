#!/usr/bin/env python.

from collections import Counter
import re
import numpy as np
import pandas as pd

###### This script generates /pickles/handcrafted_features.pkl ######

# For one data point
def generate_handcrafted_feature_vector(data):
    feature_vector = []

    #Feature 1: length of title
    feature_vector.append(len(data['title']))

    #Feature 2: length of title

    #Feature 3: length of title


    return feature_vector

# For all data points
def generate_feature_matrix(df):
    X_matrix = []
    for _, row in df.iterrows():
        feature_vector = generate_handcrafted_feature_vector(row)
        X_matrix.append(feature_vector)
    
    return X_matrix

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

    # generate influential words
    feature_vectors = generate_feature_matrix(df)
    print("Done with generating feature vectors.")

    print("Saving feature vectors to memory...")      
    save_vector_array(feature_vectors, df['labels'], filename="../pickles/handcrafted_features.pkl")
    print("Done with saving.")

if __name__ == "__main__":
    main()
