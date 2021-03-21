#!/usr/bin/env python.
import os
import pickle
from dotenv import load_dotenv
# import math
# import numpy as np
# import pandas as pd
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score

load_dotenv()
ROOT = os.environ.get("ROOT")
SEQUENCE_FEATURES_FILE = f"{ROOT}/pipeline/pickles/sequence_features.pkl"


def load_pickle(filename):
    with (open(SEQUENCE_FEATURES_FILE, "rb")) as file:
        data = pickle.load(file)
    print(data)
    exit(1)


# pass in an array of filepaths as strings
def combine_pickles(files):
    pass


def main():
    # Load data
    load_pickle(SEQUENCE_FEATURES_FILE)
    pass
    # Load data
    # df_X, df_y = load_pickle(filename="../pickles/handcrafted_features.pkl")
    # df_X, df_y = combine_pickles([TEXT_EMBEDDINGS_FILE, TITLE_EMBEDDINGS_FILE, WORD_COUNT_VECTORS_FILE])
    #
    # # Train-Test split
    # # [NOTE: No need to randomise as randomisation has already been done in scripts/dataframe_generator.py]
    # training_length = math.ceil(len(df_X) * 0.8)
    # X_train = df_X[:training_length]
    # y_train = df_y[:training_length]
    # X_test = df_X[training_length:]
    # y_test = df_y[training_length:]
    #
    # # convert to numpy array to fit into the NN
    # X_train = np.asarray(X_train)
    # y_train = np.asarray(y_train)
    # X_test = np.asarray(X_test)
    # y_test = np.asarray(y_test)
    #
    # # Training
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=(774,)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     # tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(32, activation='relu'),
    #     tf.keras.layers.Dense(3, activation='softmax')
    # ])
    #
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    # y_train = y_train.astype('int')
    # print("X_train length: ", len(X_train))
    # # print("10 examples of X_train: ", X_train[:10])
    # print("X_train feature length: ", len(X_train[0]))
    # print("y_train length: ", len(y_train))
    # print("Training model now...")
    # model.fit(X_train, y_train, epochs=5)
    #
    # # Testing
    # print("Testing model now...")
    # y_test = y_test.astype('int')
    # model.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()
