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
parser.add_argument('--input-dataset-path', type=str, help='Path to the preprocessed dataset')
parser.add_argument('--output-model-path', type=str, help='Path to the trained model')
args = parser.parse_args()

# Get dataframe
df = pd.read_csv(args.input_dataset_path, dtype={85: str})

# Get features
feature_columns = [tf.feature_column.numeric_column(key=key) for key in df.keys() if key != "Label" ]
# Get labels
labels = ["BENIGN", "Syn", "UDPLag", "UDP", "LDAP", "MSSQL", "NetBIOS", "WebDDoS"]
# Instantiate the model
classifier = tf.estimator.DNNClassifier(
        hidden_units=[60, 30, 20],
        feature_columns=feature_columns,
        n_classes=len(labels),
        label_vocabulary=labels,
        batch_norm=True,
        optimizer=lambda: tf.keras.optimizers.Adam(
            learning_rate=tf.compat.v1.train.exponential_decay(
                learning_rate=0.1,
                global_step=tf.compat.v1.train.get_global_step(),
                decay_steps=10000,
                decay_rate=0.96)
        )
)

def input_fn(df, batch_size=32):
    '''
    An input function for training or evaluating
    '''
    # Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(df[[ x for x in list(df.columns.values) if x != "Label" ]]), df["Label"]))
    # Shuffle and repeat if you are in training mode
    dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

# Train the model
logging.debug("Training model...")
chunk_size = 10**7
train_dfs = np.split(df, range(chunk_size, math.ceil(df.shape[0] / chunk_size) * chunk_size, chunk_size))
del df
round = len(train_dfs)
for train_df in train_dfs:
    logging.debug("     > Rounds to do: %i", round)
    classifier.train(input_fn=lambda: input_fn(train_df), steps=10**4)
    round -= 1

# Creating the directory where the output file will be created (the directory may or may not exist).
Path(args.output_model_path).parent.mkdir(parents=True, exist_ok=True)

# Save the model
logging.debug("Saving model...")
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(tf.feature_column.make_parse_example_spec(feature_columns))
estimator_path = classifier.export_saved_model(export_dir_base=args.output_model_path, serving_input_receiver_fn=serving_input_fn, as_text=True)
