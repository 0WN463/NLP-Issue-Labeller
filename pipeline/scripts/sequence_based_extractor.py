#!/usr/bin/env python.

import re
import numpy as np
import pandas as pd
import pickle

###### This script generates /pickles/sequence_features.pkl ######
# Format of the pickle is an array of arrays of length 3 [title, text, label]

def remove_markdown(sentence):
    markdown_pattern = r'#+|[*]+|[_]+|[>]+|[-][-]+|[+]|[`]+|!\[.+\]\(.+\)|\[.+\]\(.+\)|<.{0,6}>|\n|\r|<!---|-->|<>|=+'
    text = re.sub(markdown_pattern, ' ', sentence)
    return text

def load_dataframe_from_pickle():
    retrieved_df = pd.read_pickle("../pickles/dataframe.pkl")
    return retrieved_df

def main():
    df = load_dataframe_from_pickle()
    print("Done loading dataframe.")

    # Removing Markdown
    results = []
    for _, row in df.iterrows():
        results.append([remove_markdown(row['title']), remove_markdown(row['text']), row['labels']])
    print("Done with removing Markdown.")

    print("Saving to pickle...")      
    filename = "../pickles/sequence_features.pkl"
    outfile = open(filename, 'wb')
    pickle.dump(results, outfile)
    outfile.close()
    print("Done with pickling.")

if __name__ == "__main__":
    main()
