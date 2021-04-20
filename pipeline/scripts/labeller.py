import datetime
import os
import time

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from utils import remove_code_block, remove_url

load_dotenv()
ROOT = os.environ.get("ROOT")
SECRET_TOKEN = os.environ.get("GITHUB_TOKEN")
REPOSITORY = os.environ.get("REPOSITORY")

options = {
    "load_dir": f"{ROOT}/final_model",
    "confidence": 2,  # [0, 2, 4]. Threshold for logit output. 0 is equivalent to argmax.
    "device": torch.device("cpu"),  # cpu, cuda
}

LABELS = ["feature", "bug", "documentation", "others"]
TOKENIZER = DistilBertTokenizerFast.from_pretrained(options["load_dir"])
MODEL = DistilBertForSequenceClassification.from_pretrained(options["load_dir"], num_labels=3)
MODEL.eval()


def preprocess(text):
    preprocessed_text = remove_url(remove_code_block(text))
    return preprocessed_text


def classify(title, body):
    ''' Returns one of "bug", "documentation", "feature" '''
    preprocessed_text = preprocess(f"{title} {body}")
    encodings = TOKENIZER(preprocessed_text, truncation=True, padding=True, return_tensors='pt')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    outputs = MODEL(input_ids, attention_mask=attention_mask)
    pred = outputs["logits"].detach().numpy()
    pred = np.argmax(pred) if np.amax(pred) > options["confidence"] else -1

    return LABELS[pred]


s = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
    "Authorization": "token " + SECRET_TOKEN,
    "Accept": "application/vnd.github.v3+json"
}


def add_label(issue_num, label):
    res = s.put(f'https://api.github.com/repos/{REPOSITORY}/issues/{issue_num}/labels', headers=headers,
                json={"labels": [label]})


def main():
    print("Starting labeller...")

    ## For some reasons, when set to now(), the REST API returns nothing.
    ## So we add some interval
    payload = {
        "since": (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat()
    }

    past_issues = set()

    while True:
        res = s.get(f'https://api.github.com/repos/{REPOSITORY}/issues', headers=headers, params=payload)
        time.sleep(3)

        res = res.json()
        if len(res):
            for issue in res:
                issue_num = int(issue['url'].split('/')[-1])

                if issue_num not in past_issues:
                    label = classify(issue['title'], issue['body'])
                    add_label(issue_num, label)
                    print(f"Classified issue {issue_num} as {label}.")
                    past_issues.add(issue_num)


if __name__ == "__main__":
    main()
