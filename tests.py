import os
from random import sample
from time import time
import mne
from mne.preprocessing import ICA
from mne.decoding import CSP

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_classification

from sklearn.svm import SVC
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

import numpy as np
import matplotlib.pyplot as plt


from preprocessing import *
from utils import *


DIR_NAME = os.path.dirname(__file__)
EDF_FILES_DIR = "data"
RUNS_BASELINE = [1, 2]
RUNS_LEFT_OR_RIGHT_FIST = [3, 4, 7, 8, 11, 12]
RUNS_BOTH_FISTS_OR_FEET = [5, 6, 9, 10, 13, 14]

LABELS = {
    '01': {
        'T0': 0,
        'T1': 1
    },
    '02': {
        'T0': 0,
        'T2': 2
    },
    '12': {
        'T1': 1,
        'T2': 2
    },
    '012': {
        'T0': 0,
        'T1': 1,
        'T2': 2
    },
    # '01234': {
    #     'T0': 0,
    #     'T1': 1,
    #     'T2': 2,
    #     'T3': 3,
    #     'T4': 4,
    # }
}


def fetch_data(runs_idx: list = range(1, 15), verbose=False):
    """
        Fetch data as raw from an edf file
    """
    dir_path = os.path.join(DIR_NAME, EDF_FILES_DIR)
    dir_content_path = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

    # both_exp = any(id in RUNS_LEFT_OR_RIGHT_FIST for id in runs_idx) and any(id in RUNS_BOTH_FISTS_OR_FEET for id in runs_idx)

    raw = []
    for f in dir_content_path:
        if os.path.isfile(f):

            name, ext = os.path.splitext(f)
            _, run_id = name.split('R')

            print(f"Obj: {f} ->", run_id, ext, runs_idx)

            if ext == '.edf' and int(run_id) in runs_idx:
                print(f"EDF file: {f}")
                raw.append(mne.io.read_raw_edf(f, preload=True))

    if raw:
        raw = mne.concatenate_raws(raw)
        if verbose:
            print_raw_properties(raw)
    else:
        raw = None

    return raw


def create_pipeline():
    """create_pipeline"""

    pipeline = Pipeline([
        ('transform', StandardScaler()),
        # ('transform', ICA()),
        ('eval', LogisticRegression(multi_class='multinomial', max_iter=500))
    ])
    return pipeline


def evaluation(pipeline, X, y):
    """evaluation"""

    print(f"Evaluate dataset", X.shape, y.shape, "with pipeline", pipeline)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=int(time()))

    metric = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)
    acc = np.mean(metric)
    std = np.std(metric)

    print('Accuracy: %.3f (kfold std %.3f)' % (acc, std))
    return round(acc, 3)


if __name__ == "__main__":
    """w
    """

    # raw = fetch_data(runs_idx=RUNS_BASELINE, verbose=True)
    # raw = fetch_data(runs_idx=RUNS_LEFT_OR_RIGHT_FIST, verbose=False) # loss=0.767
    raw = fetch_data(runs_idx=RUNS_BASELINE + RUNS_LEFT_OR_RIGHT_FIST, verbose=False) # loss=0.797
    # raw = fetch_data(runs_idx=RUNS_BOTH_FISTS_OR_FEET, verbose=False)

    # raw.plot_psd(fmax=80)
    # raw.plot(n_channels=64, block=True)
    pipeline = create_pipeline()

    # X, y = make_classification(n_samples=42000, n_features=64, n_classes=3, n_clusters_per_class=1)
    accuracies = {}
    # for k, labels in LABELS.items():
    #     _, X, y = preprocessing_data(raw, labels, verbose=False)
    #     accuracies[k] = evaluation(pipeline, X, y)

    # for baseline_mode in ["mean", "ratio", "logratio", "percent", "zscore", "zlogratio"]:
    #     _, X, y = preprocessing_data(raw, LABELS['012'], baseline_mode=baseline_mode, verbose=False)
    #     loss = evaluation(pipeline, X, y)
    #     print(f"loss: {baseline_mode}: ", loss)
    #     accuracies['012' + baseline_mode] = loss

    _, X, y = preprocessing_data(raw, LABELS['012'], verbose=False)
    accuracies['012'] = evaluation(pipeline, X, y)

    print(f"Accuracies: {accuracies}")
