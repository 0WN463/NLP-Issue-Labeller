import json
import requests
import time
import datetime

def classify(title, body):
    ''' Returns one of "bug", "documentation", "feature" '''
    return "documentation"


s = requests.Session()

with open('labeller.key', 'r') as f:
    SECRET_TOKEN = f.read().strip()

headers = {
    "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
    "Authorization": "token " + SECRET_TOKEN,
    "Accept": "application/vnd.github.v3+json"
}

def add_label(issue_num, label):
    res = s.put(f'https://api.github.com/repos/0WN463/CS4248-Project/issues/{issue_num}/labels', headers=headers, json={"labels": [label]})


## For some reasons, when set to now(), the REST API returns nothing.
## So we add some interval
payload = {
    "since": (datetime.datetime.now()-  datetime.timedelta(days=1)).isoformat() 
}

past_issues = set()

while True:
    res = s.get(f'https://api.github.com/repos/0WN463/CS4248-Project/issues', headers=headers, params=payload)
    time.sleep(3)

    res = res.json()
    if len(res):
        for issue in res:
            issue_num = int(issue['url'].split('/')[-1])
            
            if issue_num not in past_issues:
                print("new issues\a", issue_num)

                label = classify(issue['title'], issue['body'])
                add_label(issue_num, label)
                past_issues.add(issue_num)
