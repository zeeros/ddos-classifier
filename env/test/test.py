import argparse
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import logging
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.DEBUG)

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Testing component for the DDoS classifier')
parser.add_argument('--input-dataset-path', type=str, help='Path to the preprocessed dataset')
parser.add_argument('--input-model-path', type=str, help='Path to the trained model')
args = parser.parse_args()

logging.debug("Testing model...")

# Get dataframe
df = pd.read_csv(args.input_dataset_path, dtype={85: str})

# Display folder content
import os
logging.debug("Model path")
modelpath = args.input_model_path.replace("inputs","utputs")
logging.debug(modelpath)

model = tf.saved_model.load(modelpath)

def predict(model, df):
  """
  returns the predicted label given a dataframe of features
  """
  feature = {k: tf.train.Feature(float_list=tf.train.FloatList(value=[v])) for k, v in dict(df).items()}
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  prediction = model.signatures["predict"](examples=tf.constant([example.SerializeToString()]))
  return prediction["classes"].numpy()[0][0].decode("utf-8")


features_name = [feature for feature in list(df) if feature != "Label"]
# Test the model
logging.debug("Testing model...")
# Set random seed for reproducibility
random.seed(9)

y_test = []
y_pred = []

for i in random.sample(range(0, len(df)), k=10**3):
  y_test.append(df['Label'].iloc[i])
  y_pred.append(predict(model, df[features_name].iloc[i]))

metrics = sklearn.metrics.classification_report(y_test, y_pred)

logging.debug("Metrics")
logging.debug(metrics)
