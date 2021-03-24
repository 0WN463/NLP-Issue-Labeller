#!/usr/bin/env python.

import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
ROOT = os.environ.get("ROOT")
DEVICE = "cpu"  # "cpu/cuda"
MODEL = 'distilbert-base-nli-stsb-mean-tokens'


###### This script generates /pickles/sentence_embedding.pkl ######
# Average of sentence embeddings

def remove_markdown(sentence):
    markdown_pattern = r'#+|[*]+|[_]+|[>]+|[-][-]+|[+]|[`]+|!\[.+\]\(.+\)|\[.+\]\(.+\)|<.{0,6}>|\n|\r|<!---|-->|<>|=+'
    text = re.sub(markdown_pattern, ' ', sentence)
    return text


def load_dataframe_from_pickle():
    retrieved_df = pd.read_pickle(f"{ROOT}/pipeline/pickles/dataframe.pkl")
    return retrieved_df


def avg_sentence_embedding(paragraph, model):
    """ Returns average of the sentence embedding vectors in the paragraph. """
    all_embeddings = []
    for sentence in nltk.sent_tokenize(paragraph):
        all_embeddings.append(model.encode(sentence))
    avg_embedding_np = np.mean(np.array(all_embeddings), axis=0)
    return avg_embedding_np


def main():
    df = load_dataframe_from_pickle()
    print("Done loading dataframe.")

    # Removing Markdown
    df["title"] = df["title"].apply(lambda x: remove_markdown(x))
    df["text"] = df["text"].apply(lambda x: remove_markdown(x))
    print("Done with removing Markdown.")

    model = SentenceTransformer(MODEL, device=DEVICE)
    title_embeddings_np = model.encode(df["title"].to_numpy())
    text_embeddings = []
    for _, para in df["text"]:
        text_embeddings.append(avg_sentence_embedding(para, model))
    text_embeddings_np = np.array(text_embeddings)
    sent_embeddings = np.concatenate((title_embeddings_np, text_embeddings_np), axis=1)
    print("Done with sentence embeddings.")

    print("Saving to pickle...")
    filename = f"{ROOT}/pipeline/pickles/sentence_embedding.pkl"
    outfile = open(filename, 'wb')
    pickle.dump(sent_embeddings, outfile)
    outfile.close()
    print("Done with pickling.")


if __name__ == "__main__":
    main()
