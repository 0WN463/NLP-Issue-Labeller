import math
import os
import pickle
import re
import shutil

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score

load_dotenv()
ROOT = os.environ.get("ROOT")

LOAD_TRAIN_PATH = f"{ROOT}/pipeline/pickles/dataframe_train.pkl"
LOAD_TEST_PATH = f"{ROOT}/pipeline/pickles/dataframe_test.pkl"
SAVE_DIR = f"{ROOT}/results/regex"
TRAIN_TEST_SPLIT = 0.8
FEATURES = ['title', 'body']

def load_pickle(filename):
    with (open(filename, "rb")) as file:
        data = pickle.load(file)
    return data

def load_dataframe_from_pickle(path):
    retrieved_df = pd.read_pickle(path)
    return retrieved_df

def bug_regex():
    ''' Returns regex to detect bug class. '''
    key_words = "(version|packages|line|file|model|core|import|source|local|device|error|build|return|unknown|backtrace|debug|bug|panic|test|what)"

    return key_words

def docs_regex():
    ''' Returns regex to detect doc class. '''
    key_words = "(issue|doc|example|version|define|model|guide|use|src|source|need|description|link|changing|api|)"

    return key_words

def features_regex():
    ''' Returns regex to detect feature class. '''
    key_words = "(feature|version|current|using|model|contrib|operation|type|would|use|unsupported|convert|information|system)"

    return key_words

def compute_regex_class(sentence):
    """ Returns class label for sentence. """
    bug = bug_regex()
    docs = docs_regex()
    features = features_regex()
    count = [len(re.findall(bug, sentence, re.IGNORECASE)),
             len(re.findall(docs, sentence, re.IGNORECASE)),
             len(re.findall(features, sentence, re.IGNORECASE))]  # bug, doc, feature counts
    if max(count) == 0:
        return 0
    else:
        return count.index(max(count))

def main():
    print("Preparing data...")
    # Load data
    train_data = load_dataframe_from_pickle(LOAD_TRAIN_PATH)
    training_length = math.ceil(len(train_data.index) * TRAIN_TEST_SPLIT)
    train_data = train_data[training_length:]  # No need training set for regex
    test_data = load_dataframe_from_pickle(LOAD_TEST_PATH)
    datasets = [train_data, test_data]

    # Retrieve features
    print("Retrieving features...")
    for ds in datasets:
        ds['X'] = ""
        for feature in FEATURES:
            ds['X'] += ds[feature] + " "

    # Regex matching
    print("Matching regex...")
    scores = []
    for ds in datasets:
        ds["pred"] = ds['X'].apply(compute_regex_class)
        Y_pred_np = ds["pred"].to_numpy()
        Y_np = ds["labels"].to_numpy()
        scores.append(accuracy_score(Y_np, Y_pred_np))

    # saving results and model
    print("Saving the good stuff...")
    info = {
        "Accuracy for seen repos": scores[0],
        "Accuracy for unseen repos": scores[1],
        "Bug regex": bug_regex(),
        "Doc regex": docs_regex(),
        "Feature regex": features_regex(),
    }

    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)  # start with clean slate
    os.makedirs(SAVE_DIR)
    data_file = open(f'{SAVE_DIR}/data.txt', "w+")
    data_file.write(str(info))
    data_file.close()


if __name__ == "__main__":
    main()
