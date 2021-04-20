# NLP Issue Labeller
Automatically labelling your github issues with NLP! Code accompanying our [paper](LINK).

![Demo](./resources/demo.gif)

## Quickstart
We use python version 3.6.5. 
1. Download the final model [https://drive.google.com/drive/u/1/folders/1Eiz4iG6SduEoUCJEIqcPg7-K-Vw7nJ7j](here).
2. Create a `.env` file in the root repository with the following details.
```
ROOT = 
GITHUB_TOKEN = 
REPOSITORY = 0WN463/CS4248-Project
```

3. Run the following steps to start labelling!
```
pip install -r requirement.txt
python pipeline/scripts/labeller.py
```
