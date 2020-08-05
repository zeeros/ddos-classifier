import argparse
import pandas as pd
import tensorflow as tf
import zipfile
import json
import collections
import logging
import preprocess
from pathlib import Path
import time
import datetime

logging.basicConfig(level=logging.DEBUG)
start = time.time()
logging.debug('START: {t}'.format(t=datetime.datetime.now()))

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Preprocessing for testing component')
parser.add_argument('--output-dataset-path', type=str, help='Path to the preprocessed testing dataset')
args = parser.parse_args()

# Creating the directory where the output file will be created (the directory may or may not exist).
Path(args.output_dataset_path).parent.mkdir(parents=True, exist_ok=True)

df = preprocess.load_dataset(
    zipfile_path = "/usr/src/app/data/CSV-03-11.zip",
    metadata_path = "/usr/src/app/data/metadata.json",
    csvs = ['Syn.csv', 'UDPLag.csv', 'UDP.csv', 'LDAP.csv', 'MSSQL.csv', 'NetBIOS.csv'],
    chunksize=8**6
)

with open(args.output_dataset_path, 'w') as dataset_file:
    df.to_csv(dataset_file, index=False)

end = time.time()
elapsed_time = (end - start)
logging.debug('Preprocessing time: {t}'.format(t=datetime.timedelta(seconds=elapsed_time)))
logging.debug('END: {t}'.format(t=datetime.datetime.now()))
