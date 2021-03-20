#!/usr/bin/env python.

from collections import Counter
import os
import math
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# import fasttext.util # for embeddings
# if not os.path.isfile('./cc.en.300.bin'):
#     # download the pre-trained model if it's not already downloaded.
#     fasttext.util.download_model('en', if_exists='ignore')  # English
# print("loading model now")
# embeddings_model = fasttext.load_model('cc.en.300.bin')
# print("done loading model")
##### Discarded: try training on our own corpus ########
# ft = fasttext.train_unsupervised('./corpus.txt', model='skipgram')

from gensim.models import Word2Vec
embeddings_model = None

LABELS = {
    "feature": 0,
    "bug": 1,
    "doc": 2
}

def generate_handcrafted_features(sentence_matrix):
    def count_word_occurrences(text, word_list):
        # remove words that appear in word_list from text
        lowercase_word_list = []
        for word in word_list:
            lowercase_word_list.append(word.lower())

        tokens = re.findall(r'\w+', text)
        count = 0
        for token in tokens:
            if token.lower() in lowercase_word_list:
                count += 1
        return count        

    processed_X = []
    for corpus in sentence_matrix:
        vector = []
        # 1. count of first and second personal pronouns (I/me, you, we/us)
        list_of_pronouns = ["i", "me", "you", "we", "us"]
        count_of_first_second_pronouns = count_word_occurrences(corpus, list_of_pronouns)
        vector.append(count_of_first_second_pronouns)

        processed_X.append(vector)
    
    return processed_X

# Given an array of sentences, transform each sentence into a vector
# representing the average of normalised embeddings of all the words.
def generate_averaged_word_embeddings(sentences):
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
        # print("here")
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
            print("done with this many sentences: ", num_of_sentences)

    return res

def generate_word_count_features(sentence_matrix, influential_words=[]):
    processed_X = []
    for corpus in sentence_matrix:
        vector = []
        tokens = re.findall(r'\w+', corpus)

        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        counter = Counter(tokens)
        for word in influential_words:
            if word.lower() in counter:
                vector.append(counter[word.lower()])
            else:
                vector.append(0)
        
        # print("vector is ", vector)
        processed_X.append(vector)
    
    return processed_X

# Removes all occurrences of words in `target` list from the `words` list.
def remove_words_from(words, target):
    res = []
    for word in words:
        if not (word in target):
            res.append(word)
    return res

# given a String of sentence, return an array of tokens
def tokeniser(sentence):
    pattern = r'\w+'
    return re.findall(pattern, sentence)

# Returns a set of n most frequent words in each category excluding m common most frequent words
# in the combined corpus.
def top_words_in_categories(n, m, sentences, labels):
    feature_examples = []
    bug_examples = []
    doc_examples = []
    for index, example in enumerate(sentences):
        if labels[index] == LABELS['feature']:
            feature_examples.append(example)
        elif labels[index] == LABELS['bug']:
            bug_examples.append(example)
        elif labels[index] == LABELS['doc']:
            doc_examples.append(example)
        else:
            print("should not reach here")

    counter = Counter(re.findall(r'\w+', " ".join(sentences)))
    most_common = counter.most_common(m)
    most_common = list(map(lambda x: x[0], most_common))
    print("most frequent common words are: ", most_common)

    feature_tokens = re.findall(r'\w+', " ".join(feature_examples))
    feature_tokens_without_common_words = remove_words_from(feature_tokens, most_common)
    counter_feature = Counter(feature_tokens_without_common_words)
    most_common_feature = counter_feature.most_common(n)
    most_common_feature = list(map(lambda x: x[0], most_common_feature))
    print("most frequent FEATURE words are: ", most_common_feature)

    bug_tokens = re.findall(r'\w+', " ".join(bug_examples))
    bug_tokens_without_common_words = remove_words_from(bug_tokens, most_common)
    counter_bug = Counter(bug_tokens_without_common_words)
    most_common_bug = counter_bug.most_common(n)
    most_common_bug = list(map(lambda x: x[0], most_common_bug))
    print("most frequent BUG words are: ", most_common_bug)

    doc_tokens = re.findall(r'\w+', " ".join(doc_examples))
    doc_tokens_without_common_words = remove_words_from(doc_tokens, most_common)
    counter_doc = Counter(doc_tokens_without_common_words)
    most_common_doc = counter_doc.most_common(n)
    most_common_doc = list(map(lambda x: x[0], most_common_doc))
    print("most frequent DOC words are: ", most_common_doc)

    return set(most_common_feature + most_common_bug + most_common_doc)

def save_vector_array(vector_array, labels):
    save_df = pd.DataFrame(columns=['Feature', 'Label'])
    # save_df = save_df.astype({'Feature': 'float64', 'Label': 'int'})
    save_df['Feature'] = pd.Series(vector_array)
    save_df['Label'] = labels.values
    save_df.to_pickle("./embeddings_pickle.pkl")

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def predict(model, X_test):
    return model.predict(X_test)

def standardise_df_labels(df, feature_labels, bug_labels, doc_labels):
    for _, row in df.iterrows():
        if row['labels'] in feature_labels:
            row['labels'] = LABELS['feature']
        elif row['labels'] in bug_labels:
            row['labels'] = LABELS['bug']
        elif row['labels'] in doc_labels:
            row['labels'] = LABELS['doc']
        else:
            print('should not reach here')
            print(row['labels'])
    df['labels'] = df['labels'].astype('int')
    return df

def main():
    # load data
    df_tensorflow = pd.read_json('../data/eng_labelled/code_text_split/tensorflow_text_code_split.json')
    df_rust = pd.read_json('../data/eng_labelled/code_text_split/rust_text_code_split.json')
    df_kubernetes = pd.read_json('../data/eng_labelled/code_text_split/kubernetes_text_code_split.json')
    
    # preprocess dataframes
    df_tensorflow = standardise_df_labels(df_tensorflow, feature_labels=['type:feature'], 
        bug_labels=['type:bug', 'type:build/install', 'type:performance', 'type:support'],
        doc_labels=['type:docs-feature', 'type:docs-bug'])
    # remove Rust "C-discussion" label
    df_rust = df_rust[df_rust.labels != 'C-discussion']
    df_rust = standardise_df_labels(df_rust, feature_labels=['C-feature-request', 'C-feature-accepted', 'C-enhancement'], 
        bug_labels=['C-bug'],
        doc_labels=['T-doc'])
    # remove Kubernetes "kind/support" label
    df_kubernetes = df_kubernetes[df_kubernetes.labels != 'kind/support']
    df_kubernetes = standardise_df_labels(df_kubernetes, feature_labels=['kind/feature', 'kind/api-change'], 
        bug_labels=['kind/bug', 'kind/failing-test'],
        doc_labels=['kind/documentation'])
    
    combined_df = pd.concat([df_tensorflow, df_rust, df_kubernetes], ignore_index=True)

    USE_PICKLED_EMBEDDINGS = False
    X_train, y_train, X_test, y_test = None, None, None, None
    training_length = math.ceil(len(combined_df) * 0.8)

    if USE_PICKLED_EMBEDDINGS:
        retrieved_df = pd.read_pickle("./embeddings_pickle.pkl")

        # manually converting Feature array to suitable format due to some errors at model.fit
        raw_X = retrieved_df['Feature']
        processed_X = []
        for row in raw_X:
            feature_vector = []
            for value in row:
                feature_vector.append(float(value))
            processed_X.append(feature_vector)

        X_train = processed_X[:training_length]
        y_train = retrieved_df['Label'][:training_length]
        X_test = processed_X[training_length:]
        y_test = retrieved_df['Label'][training_length:]

    else:
        ##### train-test split #######
        combined_df = combined_df.sample(frac=1) # randomise order

        # X_train = combined_df['text'][:training_length]
        X_train = combined_df['title'][:training_length]
        y_train = combined_df['labels'][:training_length]
        # X_test = combined_df['text'][training_length:]
        X_test = combined_df['title'][training_length:]
        y_test = combined_df['labels'][training_length:]

        print("done preprocessing dataframe")
        print("generating embeddings 1")
        sentences = []
        for _, row in combined_df.iterrows():
            sentences.append(tokeniser(row['text']))
        print("generating embeddings 2")
        global embeddings_model
        embeddings_model = Word2Vec(sentences=sentences, size=300, window=5)
        print("done with embeddings training")

        # vector_array = generate_averaged_word_embeddings(combined_df['text'])
        vector_array = generate_averaged_word_embeddings(combined_df['title'])
        print("done with embedding vectorisation")

        print("saving vector array to memory...")
        save_vector_array(vector_array, combined_df['labels'])
        print("done with saving")

        X_train = vector_array[:training_length]
        X_test = vector_array[training_length:]

    ######### Generate Influential Words ##########
    # print("Generating embeddings...")
    # influential_words = list(top_words_in_categories(150, 50, combined_df['text'], combined_df['labels']))
    # word_count_vectors = generate_word_count_features(combined_df['text'], influential_words)
    # X_train = word_count_vectors[:training_length]
    # X_test = word_count_vectors[training_length:]
    # y_train = combined_df['labels'][:training_length]
    # y_test = combined_df['labels'][training_length:]


    # processed_X_train = []
    # for i in range(len(X_train)):
    #     processed_X_train.append(processed_X_train_first_half[i].tolist() + processed_X_train_second_half[i])
    # print("Length of each vector: ", len(processed_X_train[0]))
    
    model = LogisticRegression(C=1.3, max_iter=500)
    print("Training model...")
    print("X_train length: ", len(X_train))
    print("10 examples of X_train: ", X_train[:10])
    print("X_train feature length: ", len(X_train[0]))
    print("y_train length: ", len(y_train))
    y_train = y_train.astype('int')
    train_model(model, X_train, y_train)

    # ###### validation #######
    # # processed_X_test_first_half = generate_averaged_word_embeddings(X_test)
    # # processed_X_test_second_half = generate_word_count_features(X_test, influential_words)
    # # processed_X_test = []
    # # for i in range(len(X_test)):
    # #     processed_X_test.append(processed_X_test_first_half[i].tolist() + processed_X_test_second_half[i])

    print("Testing model now...")
    y_pred = predict(model, X_test)

    # Use accurracy as the metric
    y_test = y_test.astype('int')
    score = accuracy_score(y_test, y_pred)
    print('accurracy score on validation = {}'.format(score))

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
