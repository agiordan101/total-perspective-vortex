import os
import mne

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt


DIR_NAME = os.path.dirname(__file__)
EDF_FILES_DIR = "data"


def print_raws_properties(raws):
    print(f"\n\nRaws infos:", raws)
    print(raws.info)
    print(raws.times)
    print(len(raws.times))

    print(f"\nAnnotations infos:")
    print(raws.annotations)

    events = mne.events_from_annotations(raws)
    print(f"\nEvents infos:")
    print(events)


def fetch_data(verbose=False):
    """
        Fetch data as raws from an edf file
    """
    dir_path = os.path.join(DIR_NAME, EDF_FILES_DIR)
    dir_content_path = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

    raws = []
    for f in dir_content_path:
        print(f"Obj: {f}", os.path.isfile(f), os.path.splitext(f)[1] == '.edf')
        if os.path.isfile(f) and os.path.splitext(f)[1] == '.edf':
            print(f"EDF file: {f}")
            raws.append(mne.io.read_raw_edf(f, preload=True))

    raws = mne.concatenate_raws(raws) if raws else None

    if verbose:
        print_raws_properties(raws)
    return raws

def preprocessing_data(raws, verbose=False):
    """
        Filter data

        Each annotation includes one of three codes (T0, T1, or T2):
            T0 corresponds to rest
            T1 corresponds to onset of motion (real or imagined) of
                - the left fist (in runs 3, 4, 7, 8, 11, and 12)
                - both fists (in runs 5, 6, 9, 10, 13, and 14)
            T2 corresponds to onset of motion (real or imagined) of
                - the right fist (in runs 3, 4, 7, 8, 11, and 12)
                - both feet (in runs 5, 6, 9, 10, 13, and 14)

        Transform annotations T1 for runs 5, 6, 9, 10, 13, and 14 to T3
        Transform annotations T2 for runs 5, 6, 9, 10, 13, and 14 to T4

        Goal:
            T0: rest
            T1: the left fist
            T2: the right fist
            T3: both fists
            T4: both feet
    """
    raws = raws.crop(tmax=40)

    if verbose:
        print_raws_properties(raws)
    return raws

def transform_data(raw):
    """
        Reduce dimension with CSP algorithm
    """
    csp = mne.decoding.CSP()
    return raw


# def create_pipeline():

#     cv = ShuffleSplit(10, test_size=0.2, random_state=42)
#     pipeline = Pipeline([
#         'PCA': 
#     ])


if __name__ == "__main__":
    """
        Try to classify baseline rest and both fists first
    """

    raws = fetch_data(verbose=True)

    raws = preprocessing_data(raws, verbose=True)


    # raw.plot_psd(fmax=80)
    # raw.plot(duration=5, n_channels=64, block=True)
    # print(mne.sys_info())
