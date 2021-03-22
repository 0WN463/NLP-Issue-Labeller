#!/usr/bin/env python.
import os
import pandas as pd
from dotenv import load_dotenv

###### This script generates /pickles/dataframe.pkl ######

load_dotenv()
ROOT = os.environ.get("ROOT")

LABELS = {
    "feature": 0,
    "bug": 1,
    "doc": 2
}

def standardise_df_labels(df, feature_labels, bug_labels, doc_labels):
    def standardise(label):
        if label in feature_labels:
            return LABELS['feature']
        elif label in bug_labels:
            return LABELS['bug']
        elif label in doc_labels:
            return LABELS['doc']
        else:
            print("Should not reach here.")
    df['labels'] = df['labels'].apply(standardise)
    return df

def main():
    # load data
    df_tensorflow = pd.read_json(f'{ROOT}/data/eng_labelled/code_text_split/tensorflow_text_code_split.json')
    df_rust = pd.read_json(f'{ROOT}/data/eng_labelled/code_text_split/rust_text_code_split.json')
    df_kubernetes = pd.read_json(f'{ROOT}/data/eng_labelled/code_text_split/kubernetes_text_code_split.json')
    
    # standardise dataframe labels
    df_tensorflow = standardise_df_labels(df_tensorflow, feature_labels=['type:feature'], 
        bug_labels=['type:bug', 'type:build/install', 'type:performance', 'type:support'],
        doc_labels=['type:docs-feature', 'type:docs-bug'])
    
    df_rust = df_rust[df_rust.labels != 'C-discussion'] # remove Rust "C-discussion" label
    df_rust = standardise_df_labels(df_rust, feature_labels=['C-feature-request', 'C-feature-accepted', 'C-enhancement'], 
        bug_labels=['C-bug'],
        doc_labels=['T-doc'])
    
    df_kubernetes = df_kubernetes[df_kubernetes.labels != 'kind/support'] # remove Kubernetes "kind/support" label
    df_kubernetes = standardise_df_labels(df_kubernetes, feature_labels=['kind/feature', 'kind/api-change'], 
        bug_labels=['kind/bug', 'kind/failing-test'],
        doc_labels=['kind/documentation'])
    
    combined_df = pd.concat([df_tensorflow, df_rust, df_kubernetes], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=1) # seed randomisation
    combined_df.to_pickle(f"{ROOT}/pipeline/pickles/dataframe.pkl")

if __name__ == "__main__":
    main()
