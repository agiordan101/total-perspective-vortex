import os
import mne

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt


DIR_NAME = os.path.dirname(__file__)
FILES_NAME = ['S001R01.edf', 'S001R01.edf.event']


def fetch_data(files_name: list):
    """
    Fetch data as raws from an edf file
    """
    files = [os.path.join(DIR_NAME, "data", f) for f in files_name]
    print(files)
  
    raws = []
    for f in files:
        raws.append(mne.io.read_raw_edf(f, preload=True))

    return mne.concatenate_raws(raws)


def preprocessing_data(raw):
    """ Filters"""
    pass


def transform_data(raw):
    """
        Reduce dimension with CSP algorithm
    """
    csp = mne.decoding.CSP()
    return raw


def create_pipeline():

    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        'PCA': 
    ])



if __name__ == "__main__":

    raw = fetch_data(FILES_NAME)
    events = mne.events_from_annotations(raw)

    print(raw)
    print(raw.info)
    print(raw.info["ch_names"])
    print(raw.annotations)
    print(events)

    raw.plot_psd(fmax=80)
    raw.plot(duration=5, n_channels=64, block=True)
    # print(mne.sys_info())

    # data = transform_data(raw)
    # print(data)
