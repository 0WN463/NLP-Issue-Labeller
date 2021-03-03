import requests
import json

#target_repos = ['facebook/react']
target_repos = ['rust-lang/rust']
target_repos = ['tensorflow/tensorflow']

target_labels = ['type:feature' ,'type:bug','type:docs-feature','type:docs-bug']

with open('secret.key', 'r') as f:
    SECRET_TOKEN = f.read().strip()

headers = {
    "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
    "Authorization": "token " + SECRET_TOKEN,
}

payload = {
    "state": "all",
    "per_page": 100,
}

s = requests.Session()

desired_inputs = ['title', 'body']

rate_left = int(s.get('https://api.github.com/rate_limit').json()['rate']['remaining'])
import time

def label_dic_to_string(arr):
    return ';'.join(map(lambda x: x['name'], arr))

for repo in target_repos:
    user, repo = repo.split('/')
    arr = []
    for label in target_labels:
        page = 0
        print(label)
        payload['labels'] = label
        while True:
            if rate_left == 0:
                print("Used up limit. Sleeping...")
                time.sleep(300)
                rate_left = int(s.get('https://api.github.com/rate_limit').json()['rate']['remaining'])
                continue
            payload['page'] = page
            res = s.get(f'https://api.github.com/repos/{user}/{repo}/issues', headers=headers, params=payload)
            print("requesting page", page)
            if len(res.json()) == 0:
                break
            for dic in res.json():
                if 'pull_request' not in dic:
                    data = {field: dic[field] for field in desired_inputs}
                    data['labels'] = label_dic_to_string(dic['labels'])
                    arr.append(data)
            page += 1
            rate_left -= 1

    with open(f'{repo}.json', 'w+') as f:
        f.write(json.dumps(arr, indent=4))
