#!/usr/bin/env python.

import pandas as pd

###### This script generates /pickles/dataframe.pkl ######

LABELS = {
    "feature": 0,
    "bug": 1,
    "doc": 2
}

def standardise_df_labels(df, feature_labels, bug_labels, doc_labels):
    for _, row in df.iterrows():
        if row['labels'] in feature_labels:
            row['labels'] = LABELS['feature']
        elif row['labels'] in bug_labels:
            row['labels'] = LABELS['bug']
        elif row['labels'] in doc_labels:
            row['labels'] = LABELS['doc']
        else:
            print('Should not reach here')
    df['labels'] = df['labels'].astype('int')
    return df

def main():
    # load data
    df_tensorflow = pd.read_json('../../data/eng_labelled/code_text_split/tensorflow_text_code_split.json')
    df_rust = pd.read_json('../../data/eng_labelled/code_text_split/rust_text_code_split.json')
    df_kubernetes = pd.read_json('../../data/eng_labelled/code_text_split/kubernetes_text_code_split.json')
    
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
    combined_df.to_pickle("../pickles/dataframe.pkl")

if __name__ == "__main__":
    main()
