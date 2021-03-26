import math
import os
import pickle
import re
import shutil

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score

load_dotenv()
ROOT = os.environ.get("ROOT")

SAVE_DIR = f"{ROOT}/results/regex"
SEQUENCE_FEATURES_FILE = f"{ROOT}/pipeline/pickles/sequence_features.pkl"
TRAIN_TEST_SPLIT = 0.8

"""
"regex_classifier" categorises the label based on keyword occurrences.

- keywords are chosen from the most frequent and symbolic words from each category (doc/feature/bug/other)

- word frequency analysis is done both holistically and individually on each repo

- it turns out that individual repo analysis gives us more insights into words that symbolise the category

- individual repo analysis also avoids large repo overshadows the insights from smaller repo

- frequent words that are common to many categories are not selected

- stop words and punctuations are not selected

"""

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

def other_regex():
    ''' Returns regex to detect feature class. '''
    key_words = "(master|github|version|src|name|use|cluster|node|error|service|pkg|test|code|default|file|etc|system|type|local|using|true|core|image|what|run)"

    return key_words

def load_pickle(filename):
    with (open(filename, "rb")) as file:
        data = pickle.load(file)
    return data


def main():
    print("Preparing data...")
    # Load data
    data = load_pickle(SEQUENCE_FEATURES_FILE)

    # X-Y split
    X = []
    Y = []
    for x in data:
        title, body, label = x
        # X.append(title)
        # X.append(body)
        X.append(f"{title} {body}")
        Y.append(label)

    # Test split
    # [NOTE: No need to randomise as randomisation has already been done in scripts/dataframe_generator.py]
    training_length = math.ceil(len(X) * TRAIN_TEST_SPLIT)  # no training needed for regex classifier
    X_test = X[training_length:]
    Y_test = Y[training_length:]
    Y_test_np = np.array(Y_test)

    # Regex matching
    print("Matching regex...")
    bug = bug_regex()
    docs = docs_regex()
    features = features_regex()
    other = other_regex()

    Y_pred_np = np.empty(Y_test_np.shape)
    guesses = 0
    for idx, x in enumerate(X_test):
        count = [len(re.findall(bug, x, re.IGNORECASE)),
                 len(re.findall(docs, x, re.IGNORECASE)),
                 len(re.findall(features, x, re.IGNORECASE)),
                 len(re.findall(other, x, re.IGNORECASE))]  # bug, doc, feature counts
        if max(count) == 0:
            Y_pred_np[idx] = 0  # predicts bug (most common) if there are no matches
            guesses += 1
        else:
            Y_pred_np[idx] = count.index(max(count))

    score = accuracy_score(Y_test_np, Y_pred_np)

    # saving results and model
    print("Saving the good stuff...")
    info = {
        "Accuracy": score,
        "Bug regex": bug,
        "Doc regex": docs,
        "Feature regex": features,
        "Other regex": other,
        "% Guesses": guesses / len(Y_test)
    }

    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)  # start with clean slate
    os.makedirs(SAVE_DIR)
    data_file = open(f'{SAVE_DIR}/data.txt', "w+")
    data_file.write(str(info))
    data_file.close()


if __name__ == "__main__":
    main()
