import os
import mne

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


DIR_NAME = os.path.dirname(__file__)
EDF_FILES_DIR = "data"
RUNS_REST = [1, 2]
RUNS_LEFT_OR_RIGHT_FIST = [3, 4, 7, 8, 11, 12]
RUNS_BOTH_FISTS_OR_FEET = [5, 6, 9, 10, 13, 14]
LABELS = {
    'T0': 0,
    'T1': 1,
    'T2': 2
}


def print_raws_properties(raws):
    """Display usefull stat/data
    
        Based on MNE doc (https://mne.tools/stable/auto_tutorials/epochs/40_autogenerate_metadata.html#sphx-glr-auto-tutorials-epochs-40-autogenerate-metadata-py),
            responses later than 1500 ms after stimulus onset are to be considered invalid.
        Because they don’t capture the neuronal processes of interest here.

        0.006215 ->
            160 Hz = 1 / 160 = 0.00625 s
            0.006215 to ensure all data are keep, because 160Hz is a mean
    """

    print(f"\n\nRaws infos:", raws)
    print(raws.info)
    print(raws.times)
    print(len(raws.times))

    # for ann in raws.annotations:
    #     # ann['duration'] = 1.5
    #     ann = ann.set_duration(1.5)

    print(f"\nAnnotations infos:")
    print(raws.annotations)
    print(raws.annotations[0])
    print(raws.annotations[1])
    print(len(raws.annotations))

    # Fetch event witch start each labels
    events_start_label, event_dict_start_label = mne.events_from_annotations(raws, chunk_duration=0.00625)
    print(f"\nEvents start labels:")
    print(len(events_start_label))
    print(event_dict_start_label)

    # Create annotations from those events

    # events_all, event_dict_all = mne.events_from_annotations(raws, event_id=LABELS, chunk_duration=0.006215)

    # metadata, events, events_id = mne.epochs.make_metadata(
    #     events=events, event_id=event_dict,
    #     tmin=0, tmax=0.00625, sfreq=raws.info['sfreq']
    # )
    # print(len(events))
    # print(len(events_id))
    # print(metadata)

    # metadata, events, events_id = mne.epochs.make_metadata(
    #     events=events, event_id=event_dict,
    #     tmin=0, tmax=, sfreq=raws.info['sfreq']
    # )
    # print(len(events))
    # print(len(events_id))
    # print(metadata)


    # annotations = mne.annotations_from_events(events, raws.info['sfreq'])
    # print(f"\nannotations infos:")
    # print(annotations)


def fetch_data(runs_idx: list = range(1, 15), verbose=False):
    """
        Fetch data as raws from an edf file
    """
    dir_path = os.path.join(DIR_NAME, EDF_FILES_DIR)
    dir_content_path = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

    raws = []
    for f in dir_content_path:
        if os.path.isfile(f):

            name, ext = os.path.splitext(f)
            _, run_id = name.split('R')

            print(f"Obj: {f} ->", run_id, ext, runs_idx)

            if ext == '.edf' and int(run_id) in runs_idx:
                print(f"EDF file: {f}")
                raws.append(mne.io.read_raw_edf(f, preload=True))

    if raws:
        raws = mne.concatenate_raws(raws)
        if verbose:
            print_raws_properties(raws)
    else:
        raws = None

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
    # raws = raws.crop(tmax=40)

    if verbose:
        print_raws_properties(raws)
    return raws


def transform_data(raw):
    """
        Reduce dimension with CSP algorithm
    """
    csp = mne.decoding.CSP()
    return raw


def create_pipeline():
    """create_pipeline"""

    pipeline = Pipeline([
        ('transform', StandardScaler()),
        ('classify', PCA()),
        ('eval', LogisticRegression(multi_class='multinomial'))
    ])
    return pipeline


def evaluation(pipeline, X, y):
    """evaluation"""

    print(f"Evaluate dataset", X, y, "with pipeline", pipeline)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    metric = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)

    print('Accuracy: %.3f (kfold std %.3f)' % (np.mean(metric), np.std(metric)))


if __name__ == "__main__":
    """w
    """

    # raws = fetch_data(runs_idx=RUNS_REST, verbose=True)
    raws = fetch_data(runs_idx=RUNS_LEFT_OR_RIGHT_FIST, verbose=True)
    # raws = fetch_data(runs_idx=RUNS_BOTH_FISTS_OR_FEET, verbose=True)

    # raws = preprocessing_data(raws, verbose=True)

    # raw.plot_psd(fmax=80)
    raws.plot(n_channels=64, block=True)

    # X, y = make_classification(n_samples=120000, n_features=64, n_classes=3, n_clusters_per_class=1)

    X, X_times = raws[:, :100]


    # y = X[]

    print(X.dtype)
    print(X.shape)
    # print(X_times)
    print(X_times.dtype)
    print(X_times.shape)

    print(raws.annotations)
    # plt.plot(X, y)
    # piepline = create_pipeline()
    # evaluation(piepline, X, y)
