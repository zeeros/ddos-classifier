import preprocess
import argparse
from pathlib import Path

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Preoprocessing component for the DDoS classifier')
parser.add_argument('--output-dataset-path', type=str, help='Path to the preprocessed dataset')
args = parser.parse_args()

# Creating the directory where the output file will be created (the directory may or may not exist).
Path(args.output_dataset_path).parent.mkdir(parents=True, exist_ok=True)

df = preprocess.load_dataset(
    csvs = ['UDPLag.csv', 'Syn.csv', 'DrDoS_UDP.csv', 'DrDoS_NetBIOS.csv', 'DrDoS_MSSQL.csv', 'DrDoS_LDAP.csv'],
    zipfile_path = "/usr/src/app/data/CSV-01-12.zip",
    metadata_path = "/usr/src/app/data/metadata.json",
    random_state=1
)
df = df.head(self, n=10**4)
with open(args.output_dataset_path, 'w') as dataset_file:
    df.to_csv(dataset_file)
