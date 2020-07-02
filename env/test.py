import argparse
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Training component for the DDoS classifier')
parser.add_argument('--input-model-path', type=str, help='Path to the trained model')
args = parser.parse_args()

# Load the DNN models
model = tf.saved_model.load(args.input_model_path)
