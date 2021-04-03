#!/usr/bin/env python.
import math
import os
import re

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

load_dotenv()
ROOT = os.environ.get("ROOT")

LOAD_TRAIN_PATH = f"{ROOT}/pipeline/pickles/dataframe_train.pkl"
LOAD_TEST_PATH = f"{ROOT}/pipeline/pickles/dataframe_test.pkl"
SAVE_DIR = f"{ROOT}/results/title-body"
LOAD_PATH = f"{ROOT}/results/title-body"  # load pre-trained model. If non empty, will load model instead of training from scratch.
# LOAD_PATH = None
DEVICE = torch.device("cuda:0")  # "cpu/cuda"

if LOAD_PATH:
    MODEL = DistilBertForSequenceClassification.from_pretrained(LOAD_PATH, num_labels=3)
    TOKENIZER = DistilBertTokenizerFast.from_pretrained(LOAD_PATH)
else:
    MODEL = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

options = {
    "features": ["title", "body"],  # title, body
    "train_test_split": 0.8,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 64,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "SEED": 1,
    "logging_steps": 10,
}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def remove_markdown(sentence):
    markdown_pattern = r'#+|[*]+|[_]+|[>]+|[-][-]+|[+]|[`]+|!\[.+\]\(.+\)|\[.+\]\(.+\)|<.{0,6}>|\n|\r|<!---|-->|<>|=+'
    text = re.sub(markdown_pattern, ' ', sentence)
    return text


def load_dataframe_from_pickle(path):
    retrieved_df = pd.read_pickle(path)
    return retrieved_df


def pretty_dict(dict):
    """ Returns a pretty string version of a dictionary.
    """
    result = ""
    for key, value in dict.items():
        key = str(key)
        value = str(value)
        if len(value) < 40:
            result += f'{key}: {value} \n'
        else:
            result += f'{key}: \n' \
                      f'{value} \n'
    return result


def prep_dataset(X, Y):
    """ Returns instance of Dataset class from list of training data (str) and list of labels (int). """
    encodings = TOKENIZER(X, truncation=True, padding=True)
    dataset = Dataset(encodings, Y)
    return dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


def train_model(train_dataset):
    print("Training model...")
    no_cuda = DEVICE == torch.device('cpu')
    training_args = TrainingArguments(
        output_dir=SAVE_DIR,  # output directory
        num_train_epochs=options["num_train_epochs"],  # total number of training epochs
        per_device_train_batch_size=options["per_device_train_batch_size"],  # batch size per device during training
        per_device_eval_batch_size=options["per_device_eval_batch_size"],  # batch size for evaluation
        warmup_steps=options["warmup_steps"],  # number of warmup steps for learning rate scheduler
        weight_decay=options["weight_decay"],  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=options["logging_steps"],
        no_cuda=no_cuda,
    )

    trainer = Trainer(
        model=MODEL,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        compute_metrics=compute_metrics  # accuracy metric
    )

    trainer.train()

    return trainer


def load_model():
    print("Loading model...")
    no_cuda = DEVICE == torch.device('cpu')
    training_args = TrainingArguments(  # no trg is done
        output_dir=SAVE_DIR,
        no_cuda=no_cuda,
        logging_dir='./logs',  # directory for storing logs
    )

    trainer = Trainer(
        model=MODEL,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,
        compute_metrics=compute_metrics  # accuracy metric
    )

    return trainer


def main():
    print("Preparing data...")
    # Setting seeds to control randomness
    np.random.seed(options["SEED"])
    torch.manual_seed(options["SEED"])

    # Load data
    train_data = load_dataframe_from_pickle(LOAD_TRAIN_PATH)
    test_data = load_dataframe_from_pickle(LOAD_TEST_PATH)

    # Retrieve features
    print("Retrieving features...")
    train_data['X'] = ''
    test_data['X'] = ''
    for feature in options["features"]:
        train_data['X'] += train_data[feature] + " "
        test_data['X'] += test_data[feature] + " "

    # Preprocess
    print("Preprocessing...")
    train_data['X'] = train_data['X'].apply(remove_markdown)
    test_data['X'] = test_data['X'].apply(remove_markdown)

    # Preparing model
    print("Preparing model...")
    # [NOTE: No need to randomise as randomisation has already been done in scripts/dataframe_generator.py]
    training_length = math.ceil(len(train_data.index) * options["train_test_split"])
    train_dataset = prep_dataset(train_data['X'][:training_length].tolist(),
                                 train_data['labels'][:training_length].tolist())
    test_seen_dataset = prep_dataset(train_data['X'][training_length:].tolist(),
                                     train_data['labels'][training_length:].tolist())
    test_unseen_dataset = prep_dataset(test_data['X'].tolist(), test_data['labels'].tolist())
    # lite_dataset = prep_dataset(test_data['X'][:100].tolist(),  test_data['labels'][:100].tolist())  # for dev testing

    # Building model
    load = bool(LOAD_PATH)
    if load:
        trainer = load_model()
    else:  # train from scratch
        trainer = train_model(train_dataset)
        print("Saving the good stuff in case they get lost...")
        trainer.save_model(SAVE_DIR)
        TOKENIZER.save_pretrained(SAVE_DIR)

    print("Evaluating...")
    results_seen = trainer.evaluate(test_seen_dataset)
    results_unseen = trainer.evaluate(test_unseen_dataset)
    info = options
    info["results on seen repos"] = results_seen
    info["results on unseen repos"] = results_unseen

    # saving results and model
    print("Saving all the good stuff...")
    data_file = open(f'{SAVE_DIR}/data.txt', "w+")
    data_file.write(pretty_dict(info))
    data_file.close()


if __name__ == "__main__":
    main()
