import mne

from utils import *

def preprocessing_data(raw, requested_labels, verbose=False):
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
        

        Steps:
            - set_eeg_reference
            - Filtering ?
            - Split raw in epochs, starting at annotation onset and stop 1500 ms later
            - For each epoch:
                - Create epochs
                - Labelling

    """
    print(f"\n\n\tPREPROCESSING\n")
    # raw = raw.crop(tmax=40)

    # Re-reference the data according to the average of all channels (Better classification)
    raw, _ = mne.set_eeg_reference(raw, ref_channels='average')

    # raw.plot_psd()
    # raw.filter(l_freq=0, h_freq=20)
    # raw.plot_psd()
    # print(raw)

    # Fetch event witch start each annotation/label
    events_start_label, _ = mne.events_from_annotations(raw, event_id=requested_labels)

    print(f"\tEPOCHS\n")
    # epochs = mne.make_fixed_length_epochs(raw, duration=0.5, overlap=0.1)
    epochs = mne.Epochs(raw, events_start_label, tmin=0, tmax=1.5, baseline=None)
    print(epochs)
    # mne.viz.plot_epochs(epochs, block=True)

    print(f"\tData from EPOCHS\n")
    data = epochs.get_data()                            # Shape: 3: (epochs, channels, samples)
    epochs_len, chan_len, samples_len = data.shape

    data = data.reshape((epochs_len, chan_len * samples_len))  # Shape: 2: (epochs, channels * samples)
    print(data.shape)   

    labels = events_start_label[:, 2]
    print(labels)

    if verbose:
        print_raw_properties(raw)
    return raw, data, labels


def transform_data(raw):
    """
        Reduce dimension with CSP algorithm
    """
    csp = mne.decoding.CSP()
    return raw



# def create_dataset_from_raw(raw):
#     """ Fetch data in raw
#         Create y targets with annotations
#     """

#     X, X_times = raw[:]
#     print(X.shape)
#     print(len(X.T))

#     # Fetch event witch start each annotation/label
#     events_start_label, event_dict_start_label = mne.events_from_annotations(raw, event_id=LABELS)
#     print(f"\nEvents start labels:")
#     print(len(events_start_label))

#     # Compute delta time sample for 1.5 seconds (Pertinent neuronal activity dtime)
#     dsamples_for_1500ms = int(raw.info['sfreq'] * 1.5)
#     print(f"dsamples_for_1500ms={dsamples_for_1500ms}")

#     # Attribute a label_id to each 1500ms first samples. -1 to others
#     y = np.full(len(X.T), -1)
#     for event_onset, _, event_id in events_start_label:
#         y[event_onset: event_onset + dsamples_for_1500ms] = event_id
#     print(y.shape)
#     print(y)

#     # Remove samples not within 1500ms window
#     useless_samples_idx = np.argwhere(y == -1)
#     print(f"useless_samples_idx={len(useless_samples_idx)}")

#     X = np.delete(X, useless_samples_idx, axis=1)
#     y = np.delete(y, useless_samples_idx)
#     print(X.shape)
#     print(y.shape)
#     return X.T, y
