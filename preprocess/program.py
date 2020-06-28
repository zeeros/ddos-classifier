import preprocess
import argparse
from pathlib import Path

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Preoprocessing component for the DDoS classifier')
parser.add_argument('--labels', type=List)
args = parser.parse_args()

# Creating the directory where the output file will be created (the directory may or may not exist).
#Path(args.output1_path).parent.mkdir(parents=True, exist_ok=True)

feature_columns, labels, train_dfs = preprocess.load_data(
    data_path = "/usr/src/app/data",
    train_csv = ['UDPLag.csv', 'Syn.csv', 'DrDoS_UDP.csv', 'DrDoS_NetBIOS.csv', 'DrDoS_MSSQL.csv', 'DrDoS_LDAP.csv'],
    chunk_size=9**6
)
#train_dfs.to_csv(args.output1_path)
args.labels = labels
