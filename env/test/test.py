import argparse
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import logging

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

def load_dataset(csvs, zipfile_path, metadata_path, random_state=None):

    logging.debug('Loading metadata...')
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)

    logging.debug('Opening archive...')
    archive = zipfile.ZipFile(zipfile_path, 'r')

    feature_columns = get_features(archive, metadata)

    dfs = []
    for file in archive.namelist():
        if any(file.endswith(t) for t in test_csv):
            logging.debug('     > Loading %s...', file)
            file_dfs = []
            for chunk in pd.read_csv(archive.open(file), dtype={85: str}, chunksize=chunk_size):
                df = __preprocess_dataframe(
                    chunk,
                    features=[fc.key.replace(" ", "_") for fc in feature_columns],
                    metadata=metadata
                )
                file_dfs.append({
                    "labels": df.pop("Label"),
                    "features": df
                })
            dfs.append({
                "file": file,
                "dataframe": file_dfs
            })
    return pd.concat(dfs)

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Training component for the DDoS classifier')
parser.add_argument('--input-model-path', type=str, help='Path to the trained model')
args = parser.parse_args()

# Load the DNN models
model = tf.saved_model.load(args.input_model_path)

df = load_dataset(
    csvs = ['Syn.csv', 'UDPLag.csv', 'UDP.csv', 'LDAP.csv', 'MSSQL.csv', 'NetBIOS.csv'],
    zipfile_path = "/usr/src/app/data/CSV-03-11.zip",
    metadata_path = "/usr/src/app/data/metadata.json"
)

# Test the model
metrics = []
for file_df in dfs:
  file_measures = []
  for df in file_df["dataframe"]:
    file_measures.append(model.evaluate(input_fn=lambda: input_fn(df, training=False)))
  metrics.append({
      "file": file_df["file"],
      "measures": file_measures
  })

print(metrics)
