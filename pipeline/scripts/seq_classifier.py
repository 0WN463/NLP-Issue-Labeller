#!/usr/bin/env python.
import math
import os

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

from src.utils import remove_markdown, load_dataframe_from_pickle, pretty_dict

load_dotenv()
ROOT = os.environ.get("ROOT")

LOAD_TRAIN_PATH = f"{ROOT}/pipeline/pickles/dataframe_train.pkl"
LOAD_TEST_PATH = f"{ROOT}/pipeline/pickles/dataframe_test.pkl"
SAVE_DIR = f"{ROOT}/results/title"
LOAD_PATH = f"{ROOT}/results/seq_classifier/checkpoint-8000/"  # load pre-trained model. If non empty, will load model instead of training from scratch.
# LOAD_PATH = None
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


def main():
    print("Preparing data...")
    # Setting seeds to control randomness
    np.random.seed(HP["SEED"])
    torch.manual_seed(HP["SEED"])

    # Load data
    train_data = load_dataframe_from_pickle(LOAD_TRAIN_PATH)
    test_data = load_dataframe_from_pickle(LOAD_TEST_PATH)

    # Preprocess
    train_data['X'] = (train_data['title'] + " " + train_data['body']).apply(remove_markdown)
    test_data['X'] = (test_data['title'] + " " + test_data['body']).apply(remove_markdown)

    # Preparing model
    print("Preparing model...")
    training_length = math.ceil(len(train_data.index) * HP["train_test_split"])
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_data['X'][:training_length], truncation=True, padding=True)
    test_seen_encodings = tokenizer(train_data['X'][training_length:], truncation=True, padding=True)
    test_unseen_encodings = tokenizer(test_data['X'], truncation=True, padding=True)
    # test_unseen_encodings = tokenizer(test_data['X'][:100].tolist(), truncation=True, padding=True)  # quick test

    # [NOTE: No need to randomise as randomisation has already been done in scripts/dataframe_generator.py]
    train_dataset = Dataset(train_encodings, train_data['labels'][:training_length])
    test_seen_dataset = Dataset(test_seen_encodings, train_data['labels'][training_length:])
    test_unseen_dataset = Dataset(test_unseen_encodings, test_data['labels'])
    # test_unseen_dataset = Dataset(test_unseen_encodings, test_data['labels'][:100].tolist())  # quick test

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
