import argparse
import pandas as pd
import tensorflow as tf
import zipfile
import json
import collections
import logging
import preprocess
from pathlib import Path

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Preprocessing for training component')
parser.add_argument('--output-dataset-path', type=str, help='Path to the preprocessed training dataset')
args = parser.parse_args()

# Creating the directory where the output file will be created (the directory may or may not exist).
Path(args.output_dataset_path).parent.mkdir(parents=True, exist_ok=True)

df = preprocess.load_dataset(
    zipfile_path = "/usr/src/app/data/CSV-01-12.zip",
    metadata_path = "/usr/src/app/data/metadata.json",
    random_state = 1,
    csvs = ['UDPLag.csv', 'Syn.csv', 'DrDoS_UDP.csv', 'DrDoS_NetBIOS.csv', 'DrDoS_MSSQL.csv', 'DrDoS_LDAP.csv']
)

with open(args.output_dataset_path, 'w') as dataset_file:
    df.to_csv(dataset_file, index=False)