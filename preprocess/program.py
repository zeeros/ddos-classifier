import argparse
import pandas as pd
import math
import tensorflow as tf
import zipfile
import json
import collections
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

def preprocess_dataframe(df, features, metadata=None):
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


def get_features(archive, metadata):
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
    df = preprocess_dataframe(pd.read_csv(archive.open(file), dtype={85: str}), features)
    feature_columns = []
    for key in df.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))
    return feature_columns


def load_data(data_path=".", train_csv=None, test_csv=None, chunk_size=10 ** 10):
    labels = ["BENIGN", "Syn", "UDPLag", "UDP", "LDAP", "MSSQL", "NetBIOS", "WebDDoS"]
    LoadedData = collections.namedtuple("LoadedData", "feature_columns labels train_df test_df")

    logging.debug('Load metadata')
    with open(data_path + "/metadata.json") as metadata_file:
        metadata = json.load(metadata_file)

    train_df = None
    feature_columns = None
    if train_csv is not None:
        logging.debug('Load training dataset...')
        train_archive = zipfile.ZipFile(data_path + "/CSV-01-12.zip", 'r')
        # Feature columns describe how to use the input
        feature_columns = get_features(train_archive, metadata)
        train_sets = []
        for file in train_archive.namelist():
            if any(file.endswith(t) for t in train_csv):
                logging.debug('     > Load %s', file)
                df = preprocess_dataframe(
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
        logging.debug('     Merge dataframes and shuffle')
        train_df = pd.concat(train_sets).sample(frac=1, random_state=1)

    test_dfs = None
    if test_csv is not None:
        logging.debug('Load testing dataset...')
        test_archive = zipfile.ZipFile(data_path + "/CSV-03-11.zip", 'r')
        # Feature columns describe how to use the input
        if feature_columns is None:
            feature_columns = get_features(test_archive, metadata)
        test_dfs = []
        for file in test_archive.namelist():
            if any(file.endswith(t) for t in test_csv):
                logging.debug('     > Load %s', file)
                file_test_dfs = []
                for chunk in pd.read_csv(test_archive.open(file), dtype={85: str}, chunksize=chunk_size):
                    df = preprocess_dataframe(
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

    logging.debug('Loading completed')
    return LoadedData(feature_columns, labels, train_df, test_dfs)

def load_dataset(csvs, zipfile_path, metadata_path, random_state):

    logging.debug('Loading metadata...')
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)

    logging.debug('Opening archive...')
    archive = zipfile.ZipFile(zipfile_path, 'r')

    feature_columns = get_features(archive, metadata)

    sets = []
    for file in archive.namelist():
        if any(file.endswith(t) for t in csvs):
            logging.debug('     > Loading %s...', file)
            df = preprocess_dataframe(
                df = pd.read_csv(
                    archive.open(file),
                    dtype={85: str}
                ),
                features = [fc.key.replace(" ", "_") for fc in feature_columns],
                metadata = metadata
            )
            sets.append(df)
    # Merge the dataframes into a single one and shuffle it, random_state assures reproducibility
    logging.debug('Merging and shuffling...')
    return pd.concat(sets).sample(frac=1, random_state=random_state)


# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Preoprocessing component for the DDoS classifier')
parser.add_argument('--output-dataset-path', type=str, help='Path to the preprocessed dataset')
args = parser.parse_args()

# Creating the directory where the output file will be created (the directory may or may not exist).
Path(args.output_dataset_path).parent.mkdir(parents=True, exist_ok=True)

df = load_dataset(
    csvs = ['UDPLag.csv', 'Syn.csv', 'DrDoS_UDP.csv', 'DrDoS_NetBIOS.csv', 'DrDoS_MSSQL.csv', 'DrDoS_LDAP.csv'],
    zipfile_path = "/usr/src/app/data/CSV-01-12.zip",
    metadata_path = "/usr/src/app/data/metadata.json",
    random_state = 1
)

with open(args.output_dataset_path, 'w') as dataset_file:
    df.to_csv(dataset_file, index=False)
