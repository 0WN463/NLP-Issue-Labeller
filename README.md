# CS4248 Natural Language Processing Project

## ML Model Logs
Please refer to this Google Doc [https://docs.google.com/document/d/1gbVSwV-hOVDH6KOhZyQZVdEpqOnWA63lkSm7FQlKa70/edit](https://docs.google.com/document/d/1gbVSwV-hOVDH6KOhZyQZVdEpqOnWA63lkSm7FQlKa70/edit) for current model performance.

## Note
The feature engineering and ML model files are in the ``pipeline/scripts`` directory. The pickled data
files are in ``pipeline/pickles``. Note that the pickle files are separated based on the type of features 
so that it's easier to try experimenting with and combining different features during ML training.

## Installation
Python Version: 3.6.5. 

```
pip install -r requirements.txt  # install dependencies (use virtual envs if u're comfortable!)
echo "ROOT = %path-to-proj-root-dir" >> .env  # for absolute pathing that's independent of machine  
```

If you're comfortable, you can use [pyenv](https://github.com/pyenv/pyenv) to manage python versions
and [pyvenv](https://github.com/pyenv/pyenv-virtualenv) to manage virtual environments.
