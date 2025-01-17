import argparse
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import logging
import time
import datetime


logging.basicConfig(level=logging.DEBUG)
start = time.time()
logging.debug('START: {t}'.format(t=datetime.datetime.now()))

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Training component for the DDoS classifier')
parser.add_argument('--hidden-layers', type=str, default='', help='Path to the preprocessed dataset')
parser.add_argument('--input-dataset-path', type=str, help='Path to the preprocessed dataset')
parser.add_argument('--output-model-path', type=str, help='Path to the trained model')
args = parser.parse_args()

# Get dataframe
df = pd.read_csv(args.input_dataset_path, dtype={85: str})
# Get hidden layers
if args.hidden_layers == "":
    HIDDEN_UNITS = [[60, 30, 20], [60, 40, 30, 20]]
else:
    HIDDEN_UNITS = [args.hidden_layers.split(",")]

# Get features
feature_columns = [tf.feature_column.numeric_column(key=key) for key in df.keys() if key != "Label" ]
# Get labels
labels = ["BENIGN", "Syn", "UDPLag", "UDP", "LDAP", "MSSQL", "NetBIOS", "WebDDoS"]

def input_fn(df, training, batch_size=32):
    '''
    An input function for training or evaluating
    '''
    # Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(df[[ x for x in list(df.columns.values) if x != "Label" ]]), df["Label"]))
    # Shuffle and repeat if you are in training mode
    if training:
      dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

def run_config(hparams, model_dir):
  start = time.time()
  classifier = tf.estimator.DNNClassifier(
      hidden_units=hparams['HIDDEN_UNITS'],
      feature_columns=feature_columns,
      n_classes=len(labels),
      label_vocabulary=labels,
      batch_norm=True,
      model_dir=model_dir,
      dropout=hparams['DROPOUT'],
      optimizer=lambda: tf.keras.optimizers.Adam(
          learning_rate=tf.compat.v1.train.exponential_decay(
              learning_rate=hparams['LEARNING_RATE'],
              global_step=tf.compat.v1.train.get_global_step(),
              decay_steps=10000,
              decay_rate=0.96)
      ),
      config=tf.estimator.RunConfig(
          tf_random_seed=9,
          log_step_count_steps=10**5
      )
  )
  start = time.time()
  train_df_count = int(round*(80/100))
  # Train over #train_df_count dataframes
  for train_df in train_dfs[:train_df_count]:
    classifier.train(input_fn=lambda: input_fn(train_df, training=True), steps=10**4)
  end = time.time()
  training_time = (end - start)
  start = time.time()
  accuracies = []
  # Evaluate over the remaining dataframes
  for evaluate_df in train_dfs[train_df_count:]:
    accuracies.append(classifier.evaluate(input_fn=lambda: input_fn(evaluate_df, training=False))['accuracy'])
  end = time.time()
  validation_time = (end - start)
  return {
        "hparams": hparams,
        "training_time": '{t}'.format(t=datetime.timedelta(seconds=training_time)),
        "validation_time": '{t}'.format(t=datetime.timedelta(seconds=validation_time)),
        "classifier": classifier,
        "accuracy": sum(accuracies)/len(accuracies)
    }

# Train the model
logging.debug("Training model...")
chunk_size = 9**6
train_dfs = np.split(df, range(chunk_size, math.ceil(df.shape[0] / chunk_size) * chunk_size, chunk_size))
del df
round = len(train_dfs)

session_num = 0
session_runs = []
best_run = None

DROPOUT = [0.1, 0.2]
LEARNING_RATE = [0.1, 0.3]
for hidden_units in HIDDEN_UNITS:
    for dropout in DROPOUT:
      for learning_rate in LEARNING_RATE:
        hparams = {}
        hparams['HIDDEN_UNITS'] = hidden_units
        hparams['DROPOUT'] = dropout
        hparams['LEARNING_RATE'] = learning_rate
        logging.debug("Session #%d" % session_num)
        logging.debug('hparams: %s', hparams)
        run = run_config(hparams=hparams, model_dir=args.output_model_path+"/"+str(session_num))
        logging.debug("Training time: %s", run["training_time"])
        logging.debug("Validation time: %s", run["validation_time"])
        session_runs.append(run)
        if best_run is None or best_run["accuracy"] < run["accuracy"]:
            # Set current model as the classifier to export
            best_run = run
        session_num += 1

# Train the model
logging.debug("Training end")
logging.debug("- Session runs:")
logging.debug(session_runs)
logging.debug("- Best run:")
logging.debug(best_run)

# Creating the directory where the output file will be created (the directory may or may not exist).
Path(args.output_model_path).parent.mkdir(parents=True, exist_ok=True)

# Save the model
logging.debug("Saving model...")
classifier = best_run["classifier"]
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(tf.feature_column.make_parse_example_spec(feature_columns))
estimator_path = classifier.export_saved_model(export_dir_base=args.output_model_path+"/model", serving_input_receiver_fn=serving_input_fn)
logging.debug("- Estimator path")
logging.debug(estimator_path)

end = time.time()
elapsed_time = (end - start)
logging.debug('Training time: {t}'.format(t=datetime.timedelta(seconds=elapsed_time)))
logging.debug('END: {t}'.format(t=datetime.datetime.now()))
