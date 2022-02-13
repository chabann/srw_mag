import csv
import json
import numpy as np
import warnings
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

if __name__ == "__main__":

    #with open('data/ernstandyung.vfa') as f:
        #d = json.load(f)
    d = pd.read_json('data/ernstandyung.vfa', orient='index')
    print(d)

    d.to_csv()
    print(d)

