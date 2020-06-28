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


def load_data(data_path=".", train_csv=None, test_csv=None, chunk_size=10 ** 10):
    labels = ["BENIGN", "Syn", "UDPLag", "UDP", "LDAP", "MSSQL", "NetBIOS", "WebDDoS"]
    LoadedData = collections.namedtuple("LoadedData", "feature_columns labels train_dfs test_dfs")

    logging.debug('Load metadata')
    with open(data_path + "/metadata.json") as metadata_file:
        metadata = json.load(metadata_file)

    train_dfs = None
    feature_columns = None
    if train_csv is not None:
        logging.debug('Load training dataset...')
        train_archive = zipfile.ZipFile(data_path + "/CSV-01-12.zip", 'r')
        # Feature columns describe how to use the input
        logging.debug('Get feature columns', feature_columns)
        feature_columns = __get_features(train_archive, metadata)
        logging.debug(feature_columns)
        train_sets = []
        for file in train_archive.namelist():
            if any(file.endswith(t) for t in train_csv):
                logging.debug(' > Load', file)
                df = __preprocess_dataframe(
                    df=pd.read_csv(
                        train_archive.open(file),
                        dtype={85: str}
                    ),
                    features=[fc.key.replace(" ", "_") for fc in feature_columns],
                    metadata=metadata
                )
                # Load csv to dataframe
                train_sets.append(df)
        # Merge the dataframes into a single one and shuffle it, random_state assures reproducibility
        logging.debug(' Merge dataframes and shuffle')
        train_sets = pd.concat(train_sets).sample(frac=1, random_state=1)
        # Split the dataframes in multiple chunks
        logging.debug(' Split into smaller dataframes')
        train_chunks = np.split(train_sets,
                                range(chunk_size, math.ceil(train_sets.shape[0] / chunk_size) * chunk_size, chunk_size))
        del train_sets
        train_dfs = []
        for train_chunk in train_chunks:
            train_dfs.append({
                "labels": train_chunk.pop("Label"),
                "features": train_chunk
            })
        del train_chunks

    test_dfs = None
    if test_csv is not None:
        logging.debug('Load testing dataset...')
        test_archive = zipfile.ZipFile(data_path + "/CSV-03-11.zip", 'r')
        # Feature columns describe how to use the input
        if feature_columns is None:
            feature_columns = __get_features(test_archive, metadata)
        test_dfs = []
        for file in test_archive.namelist():
            if any(file.endswith(t) for t in test_csv):
                logging.debug(' > Load', file)
                file_test_dfs = []
                for chunk in pd.read_csv(test_archive.open(file), dtype={85: str}, chunksize=chunk_size):
                    df = __preprocess_dataframe(
                        chunk,
                        features=[fc.key.replace(" ", "_") for fc in feature_columns],
                        metadata=metadata
                    )
                    file_test_dfs.append({
                        "labels": df.pop("Label"),
                        "features": df
                    })
                test_dfs.append({
                    "file": file,
                    "dataframe": file_test_dfs
                })
    logging.debug('Data is loaded')
    return LoadedData(feature_columns, labels, train_dfs, test_dfs)
