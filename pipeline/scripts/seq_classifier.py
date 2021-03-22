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

SEQUENCE_FEATURES_FILE = f"{ROOT}/pipeline/pickles/sequence_features.pkl"
SAVE_DIR = f"{ROOT}/results/seq_classifier"
LOAD_PATH = f"{ROOT}/results/seq_classifier/checkpoint-8000/"  # load pre-trained model. If non empty, will load model instead of training from scratch.
DEVICE = torch.device("cpu")  # "cpu/cuda"

HP = {  # hyperparameters
    "train_test_split": 0.999,  #0.8
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

def train_model(train_dataset, test_dataset):
    print("Training model...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,  # output directory
        num_train_epochs=HP["num_train_epochs"],  # total number of training epochs
        per_device_train_batch_size=HP["per_device_train_batch_size"],  # batch size per device during training
        per_device_eval_batch_size=HP["per_device_eval_batch_size"],  # batch size for evaluation
        warmup_steps=HP["warmup_steps"],  # number of warmup steps for learning rate scheduler
        weight_decay=HP["weight_decay"],  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=HP["logging_steps"],
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
        compute_metrics=compute_metrics  # accuracy metric
    )

    trainer.train()

    return trainer

def load_model(load_path, test_dataset):
    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(load_path, num_labels=3)
    model.to(DEVICE)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=None,  # already trained
        train_dataset=None,  
        eval_dataset=test_dataset,  # evaluation dataset
        compute_metrics=compute_metrics  # accuracy metric
    )

    return trainer

def main():
    print("Preparing data...")
    # Setting seeds to control randomness
    np.random.seed(HP["SEED"])
    torch.manual_seed(HP["SEED"])

    # Load data
    data = load_pickle(SEQUENCE_FEATURES_FILE)

    # X-Y split
    X = []
    Y = []
    for x in data:
        title, _, label = x  # TODO: try out body_text too
        X.append(title)
        Y.append(label)

    # Train-Test split
    # [NOTE: No need to randomise as randomisation has already been done in scripts/dataframe_generator.py]
    training_length = math.ceil(len(X) * HP["train_test_split"])
    X_train = X[:training_length]
    Y_train = Y[:training_length]
    X_test = X[training_length:]
    Y_test = Y[training_length:]

    # Preparing model
    print("Preparing model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    test_encodings = tokenizer(X_test, truncation=True, padding=True)

    train_dataset = Dataset(train_encodings, Y_train)
    test_dataset = Dataset(test_encodings, Y_test)

    # Building model
    load = bool(LOAD_PATH)
    if load:
        trainer = load_model(LOAD_PATH, test_dataset)
    else:  # train from scratch
        trainer = train_model(train_dataset, test_dataset)

    print("Evaluating...")
    results = trainer.evaluate()
    info = HP
    info["results"] = results

    # saving results and model
    print("Saving the good stuff...")
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    data_file = open(f'{SAVE_DIR}/data.txt', "w+")
    data_file.write(info)
    data_file.close()


if __name__ == "__main__":
    main()
