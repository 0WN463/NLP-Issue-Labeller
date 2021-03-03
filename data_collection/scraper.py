import json
import requests
import sys
import time

arr = []

for arg in sys.argv[1:]:
    with open(arg, 'r') as f:
        j = json.load(f)
        arr.append(j)

#target_repos = ['tensorflow/tensorflow']

#target_labels = [
#    'type:feature' ,
#    'type:bug',
#    'type:docs-feature',
#    'type:docs-bug',
#    'type:build/install',
#    'type:performance',
#    'type:support'
#    ]

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

for dic in arr:
    repo = dic['repo']
    print(f"Scraping {repo}...")

    target_labels = dic['desired_labels']
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
                if page == 0:
                    print("Warning: {label} returned no pages")
                break
            for dic in res.json():
                if 'pull_request' not in dic:
                    data = {field: dic[field] for field in desired_inputs}
                    data['labels'] = label 
                    arr.append(data)
            page += 1
            rate_left -= 1

    with open(f'{repo}.json', 'w+') as f:
        f.write(json.dumps(arr, indent=4))
