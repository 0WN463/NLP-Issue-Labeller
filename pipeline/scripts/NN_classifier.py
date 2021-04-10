#!/usr/bin/env python.

import os
from dotenv import load_dotenv
import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import seaborn as sn
import matplotlib.pyplot as plt

# for imbalanced dataset
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

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

def plot_confusion_matrix(y_true, y_pred):
    confusion_array = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(confusion_array, index = range(3), columns = range(3))
    # plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap='viridis_r', fmt='d')
    plt.show()

# Only class with a probability above threshold will get a classification label
def predict_with_threshold(model, X_test, threshold=0.5):
    predictions = model(X_test).numpy()
    predictions = tf.nn.softmax(predictions).numpy()
    results = []
    for row in predictions:
        if np.amax(row) >= threshold:
            results.append(np.argmax(row))
        else:
            results.append(-1) # -1 indicates no decision
    
    return results

def accuracy_labelled(pred, y_test):
    if len(pred) != len(y_test):
        print("Wrong dimensions!")
        return
    total_labelled_instances = 0
    total_correct_instances = 0
    num_doc = 0
    for i in range(len(pred)):
        if pred[i] != -1:
            total_labelled_instances += 1
            if pred[i] == y_test[i]:
                total_correct_instances += 1
            if pred[i] == 2:
                num_doc += 1
    
    print("total is ", total_labelled_instances)
    print("total correct is ", total_correct_instances)
    print("number of doc issues: ", num_doc)

    return total_correct_instances / total_labelled_instances

def main():
    # Load training data. NOTE: only pass in seen repos here
    print("Combining pickles...")
    df_X_seen, df_y_seen = combine_pickles([TEXT_EMBEDDINGS_SEEN, TITLE_EMBEDDINGS_SEEN])

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

    # RESAMPLE DATA
    # oversample = SMOTE(sampling_strategy='minority', k_neighbors=2)
    # oversample = RandomOverSampler(sampling_strategy={2 : 5000})
    oversample = RandomOverSampler(sampling_strategy='not majority', random_state=1)
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    print(sorted(Counter(y_train).items()))

    # Training
    input_length = len(X_train[0])
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_length,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
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
    plot_confusion_matrix(y_test_seen, np.argmax(model.predict(X_test_seen), axis=1))

    # Check accuracy if we set threshold above 50%
    # predictions = model(X_test_seen[:30]).numpy()
    # predictions = tf.nn.softmax(predictions).numpy()
    adjusted_pred_seen = predict_with_threshold(model, X_test_seen, threshold=0.5)
    # print("adjusted predictions:")
    # print(predictions)
    # print(adjusted_pred[:30])
    print("adjusted accuracy:")
    adjusted_accuracy_seen = accuracy_labelled(adjusted_pred_seen, y_test_seen)
    print(adjusted_accuracy_seen)

    # sanity checks
    X_test_seen = None
    y_test_seen = None

    # Testing for unseen repos
    # load unseen repos first
    df_X_unseen, df_y_unseen = combine_pickles([TEXT_EMBEDDINGS_UNSEEN, TITLE_EMBEDDINGS_UNSEEN])
    df_X_unseen = np.asarray(df_X_unseen)
    df_y_unseen = np.asarray(df_y_unseen)
    print("length of df_X_unseen: ", len(df_X_unseen)) # sanity check
    print("Testing model on UNSEEN repos now...")
    model.evaluate(df_X_unseen, df_y_unseen)
    plot_confusion_matrix(df_y_unseen, np.argmax(model.predict(df_X_unseen), axis=1))

    # Check accuracy if we set threshold above 50%
    adjusted_pred_unseen = predict_with_threshold(model, df_X_unseen, threshold=0.5)
    print("adjusted accuracy unseen:")
    adjusted_accuracy_unseen = accuracy_labelled(adjusted_pred_unseen, df_y_unseen)
    print(adjusted_accuracy_unseen)

if __name__ == "__main__":
    main()
