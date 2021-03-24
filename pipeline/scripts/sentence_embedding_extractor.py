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
INPUT_COL = "text"  # [title, text]
DEVICE = "cuda"  # "cpu/cuda"
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
    all_embeddings = model.encode(nltk.sent_tokenize(paragraph))
    avg_embedding_np = np.nanmean(all_embeddings, axis=0) if not np.isnan(all_embeddings).all() else np.zeros(768)
    return avg_embedding_np

def save_vector_array(vector_array, labels, filename):
    save_df = pd.DataFrame(columns=['Feature', 'Label'])
    save_df['Feature'] = pd.Series(vector_array)
    save_df['Label'] = labels.values
    save_df.to_pickle(filename)

def main():
    df = load_dataframe_from_pickle()
    print("Done loading dataframe.")

    # Removing Markdown
    df[INPUT_COL] = df[INPUT_COL].apply(lambda x: remove_markdown(x))
    print("Done with removing Markdown.")

    model = SentenceTransformer(MODEL, device=DEVICE)
    sent_embeddings= []  # 1-D
    if INPUT_COL == "title":
        for _, sent in df["title"].items():
            sent_embeddings.append(model.encode(sent))
    elif INPUT_COL == "text":
        for _, para in df["text"].items():
            sent_embeddings.append(avg_sentence_embedding(para, model))
    else:
        raise NotImplementedError("Only supports embedding of title and text for now.")
    print("Done with sentence embeddings.")

    print("Saving feature vectors to disc...")
    filename = f"{ROOT}/pipeline/pickles/{INPUT_COL}_sentence_embeddings.pkl"
    save_vector_array(sent_embeddings, df['labels'], filename=filename)
    print("Done with saving.")


if __name__ == "__main__":
    main()
