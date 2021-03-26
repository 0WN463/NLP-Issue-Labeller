#!/usr/bin/env python.
import math
import os
import pickle

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

load_dotenv()
ROOT = os.environ.get("ROOT")

SEQUENCE_FEATURES_TRAIN_FILE = f"{ROOT}/pipeline/pickles/sequence_features_train.pkl"
SEQUENCE_FEATURES_TEST_FILE = f"{ROOT}/pipeline/pickles/sequence_features_test.pkl"
SAVE_DIR = f"{ROOT}/results/title"
# LOAD_PATH = f"{ROOT}/results/seq_classifier/checkpoint-8000/" # load pre-trained model. If non empty, will load model instead of training from scratch.
LOAD_PATH = None
DEVICE = torch.device("cuda")  # "cpu/cuda"

HP = {  # hyperparameters
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


def load_pickle(filename):
    with (open(filename, "rb")) as file:
        data = pickle.load(file)
    return data


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

def train_model(train_dataset):
    print("Training model...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    # model.to(DEVICE)

    no_cuda = DEVICE == torch.device('cpu')
    training_args = TrainingArguments(
        output_dir=SAVE_DIR,  # output directory
        num_train_epochs=HP["num_train_epochs"],  # total number of training epochs
        per_device_train_batch_size=HP["per_device_train_batch_size"],  # batch size per device during training
        per_device_eval_batch_size=HP["per_device_eval_batch_size"],  # batch size for evaluation
        warmup_steps=HP["warmup_steps"],  # number of warmup steps for learning rate scheduler
        weight_decay=HP["weight_decay"],  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=HP["logging_steps"],
        no_cuda=no_cuda,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        compute_metrics=compute_metrics  # accuracy metric
    )

    trainer.train()

    return trainer

def load_model(load_path):
    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(load_path, num_labels=3)

    no_cuda = DEVICE == torch.device('cpu')
    training_args = TrainingArguments(  # no trg is done
        output_dir=SAVE_DIR,
        no_cuda=no_cuda,
        logging_dir='./logs',  # directory for storing logs
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,
        compute_metrics=compute_metrics  # accuracy metric
    )

    return trainer

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


def main():
    print("Preparing data...")
    # Setting seeds to control randomness
    np.random.seed(HP["SEED"])
    torch.manual_seed(HP["SEED"])

    # Load data
    train_data = load_pickle(SEQUENCE_FEATURES_TRAIN_FILE)
    test_data = load_pickle(SEQUENCE_FEATURES_TEST_FILE)
    all_data = [train_data, test_data]

    # X-Y split
    X = [[], []]  # train data, test data
    Y = [[], []]
    for i, data in enumerate(all_data):
        for x in data:
            title, body, label = x
            X[i].append(title)
            # X.append(body)
            Y[i].append(label)

    # Preparing model
    print("Preparing model...")
    training_length = math.ceil(len(X) * HP["train_test_split"])
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(X[0][:training_length], truncation=True, padding=True)
    test_seen_encodings = tokenizer(X[0][training_length:], truncation=True, padding=True)
    test_encodings = tokenizer(X[1], truncation=True, padding=True)
    # test_encodings = tokenizer(X[1][:100], truncation=True, padding=True)  # quick test

    # [NOTE: No need to randomise as randomisation has already been done in scripts/dataframe_generator.py]
    train_dataset = Dataset(train_encodings, Y[0][:training_length])
    test_seen_dataset = Dataset(test_seen_encodings, Y[0][training_length:])
    test_unseen_dataset = Dataset(test_encodings, Y[1])
    # test_unseen_dataset = Dataset(test_encodings, Y[1][:100])  # quick test

    # Building model
    load = bool(LOAD_PATH)
    if load:
        trainer = load_model(LOAD_PATH)
    else:  # train from scratch
        trainer = train_model(train_dataset)
        print("Saving the good stuff in case they get lost...")
        trainer.save_model(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)

    print("Evaluating...")
    results_seen = trainer.evaluate(test_seen_dataset)
    results_unseen = trainer.evaluate(test_unseen_dataset)
    info = HP
    info["results on seen repos"] = results_seen
    info["results on unseen repos"] = results_unseen

    # saving results and model
    print("Saving all the good stuff...")
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    data_file = open(f'{SAVE_DIR}/data.txt', "w+")
    data_file.write(pretty_dict(info))
    data_file.close()


if __name__ == "__main__":
    main()
