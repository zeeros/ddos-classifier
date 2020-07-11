import argparse
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import logging
from sklearn.metrics import *
from sklearn import metrics

logging.basicConfig(level=logging.DEBUG)

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Testing component for the DDoS classifier')
parser.add_argument('--input-dataset-path', type=str, help='Path to the preprocessed dataset')
parser.add_argument('--output-model-path', type=str, help='Path to the trained model')
parser.add_argument('--output-model-path', type=str, help='Path to the trained model')
args = parser.parse_args()

# Get dataframe
df = pd.read_csv(args.input_dataset_path, dtype={85: str})

# Get features
feature_columns = [tf.feature_column.numeric_column(key=key) for key in df.keys() if key != "Label" ]
# Get labels
labels = ["BENIGN", "Syn", "UDPLag", "UDP", "LDAP", "MSSQL", "NetBIOS", "WebDDoS"]
# Get the only folder inside the path, containing the model
input_model_path = args.input_model_path + "/" + os.listdir(args.input_model_path)[0]
# Load the DNN models
model = tf.saved_model.load(input_model_path)

# Test the model

# Creating the directory where the output file will be created (the directory may or may not exist).
Path(args.output_model_path).parent.mkdir(parents=True, exist_ok=True)

# Save the model
logging.debug("Saving model...")
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(tf.feature_column.make_parse_example_spec(feature_columns))
estimator_path = classifier.export_saved_model(export_dir_base=args.output_model_path, serving_input_receiver_fn=serving_input_fn, experimental_mode=ModeKeys.EVAL)
