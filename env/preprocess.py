import pandas as pd
import numpy as np
import tensorflow as tf
import math
import zipfile
import json
import collections
import logging


logging.basicConfig(level=logging.DEBUG)

def __preprocess_dataframe(df, features, metadata=None):
    """
    If metadata is passed, return also the labels
    """
    # Trim columns name, replace whitespaces from columns name
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))
    # Keep only features and label
    df = df[features + ["Label"]]
    if metadata is None:
        # Remove label column from features
        df.pop('Label')
    else:
        # Labels between training and set dataset may be different, map them using metadata
        labels = df['Label'].unique()
        for l in labels:
            if l in metadata['Label']['01-12->03-11']:
                df['Label'].replace(to_replace=l, value=metadata['Label']['01-12->03-11'][l], inplace=True)
    return df


def __get_features(archive, metadata):
    """
    Retrieve the features from the archive
    """
    features = []
    # Get a flat list of all best features
    for dataset_label in metadata['Label']['Features']:
        features += metadata['Label']['Features'][dataset_label]
    features = list(set(features))
    features = [fc.replace(" ", "_") for fc in features]
    # Use the first csv file in the archive to extract column features
    file = next((file for file in archive.namelist() if file.endswith(".csv")), None)
    df = __preprocess_dataframe(pd.read_csv(archive.open(file), dtype={85: str}), features)
    feature_columns = []
    for key in df.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))
    return feature_columns

def load_dataset(zipfile_path, metadata_path, random_state, csvs=None, chunksize=None, shuffle=True):

    logging.debug('Loading metadata...')
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)

    logging.debug('Opening archive...')
    archive = zipfile.ZipFile(zipfile_path, 'r')

    feature_columns = __get_features(archive, metadata)

    sets = []
    for file in archive.namelist():
        # Load all the files if csvs parameter is None
        # otherwise check that the file is listed in csvs
        if csvs is None:
            load_file = file.endswith(".csv")
        else:
            load_file = any(file.endswith(t) for t in csvs)
        if load_file:
            logging.debug('     > Loading %s...', file)
            # Load the entire csv file only if chunksize is not specified
            if chunksize is None:
                df = __preprocess_dataframe(
                    df = pd.read_csv(
                        archive.open(file),
                        dtype={85: str}
                    ),
                    features = [fc.key.replace(" ", "_") for fc in feature_columns],
                    metadata = metadata
                )
                sets.append(df)
            else:
                for chunk in pd.read_csv(archive.open(file), dtype={85: str}, chunksize=chunksize):
                    df = __preprocess_dataframe(
                        df = chunk,
                        features = [fc.key.replace(" ", "_") for fc in feature_columns],
                        metadata = metadata
                    )
                    sets.append(df)
    # Merge the dataframes into a single one and shuffle it, random_state assures reproducibility
    logging.debug('Merging and shuffling...')
    df = pd.concat(sets, ignore_index=True)
    if shuffle:
        return df.sample(frac=1, random_state=random_state)
    else:
        return df
