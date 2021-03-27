# CS4248 Natural Language Processing Project

## ML Model Logs
Please refer to this Google Doc [https://docs.google.com/document/d/1gbVSwV-hOVDH6KOhZyQZVdEpqOnWA63lkSm7FQlKa70/edit](https://docs.google.com/document/d/1gbVSwV-hOVDH6KOhZyQZVdEpqOnWA63lkSm7FQlKa70/edit) for current model performance.

## Note
1. The feature engineering and ML model files are in the ``pipeline/scripts`` directory. The pickled data files are in ``pipeline/pickles``. Note that the pickle files are separated based on the type of features so that it's easier to try experimenting with and combining different features during ML training.
2. ``pipeline/pickles/dataframe_train.pkl`` and ``pipeline/pickles/dataframe_test.pkl`` are not committed/pushed to GitHub due to their large size. You may run ``pipeline/scripts/dataframe_generator.py`` to obtain the same dataframe pickle files when you need them.

## Installation
Set your ROOT path via 

```
echo "ROOT = %path-to-proj-root-dir" >> .env  # for absolute pathing that's independent of machine  
```

To run `seq_classifer.py`, use python version 3.7+ and install based on versioning stated in `requirements.txt`. You can either install on demand (i.e. only when you need it to run a script) or install all dependencies using


```
pip install -r requirements.txt  # install dependencies
```

If you're comfortable, you can use [pyenv](https://github.com/pyenv/pyenv) to manage python versions
and [pyvenv](https://github.com/pyenv/pyenv-virtualenv) to manage virtual environments.

Feel free to add your dependencies on top of mine using `pip freeze`.
