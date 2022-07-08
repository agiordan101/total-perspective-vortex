import mne


def print_raw_properties(raw):
    """Display usefull stat/data
    
        Based on MNE doc (https://mne.tools/stable/auto_tutorials/epochs/40_autogenerate_metadata.html#sphx-glr-auto-tutorials-epochs-40-autogenerate-metadata-py),
            responses later than 1500 ms after stimulus onset are to be considered invalid.
        Because they don’t capture the neuronal processes of interest here.

        0.006215 ->
            160 Hz = 1 / 160 = 0.00625 s
            0.006215 to ensure all data are keep, because 160Hz is a mean
    """

    print(f"\n\nraw infos:", raw)
    print(raw.info)
    print(raw.times)
    print(len(raw.times))

    print(f"\nAnnotations infos:")
    print(raw.annotations)
    print(raw.annotations[0])

    # events_all, event_dict_all = mne.events_from_annotations(raw, event_id=LABELS, chunk_duration=0.006215)
    # print(f"\nAll events:")
    # print(len(events_all))
    # print(event_dict_all)