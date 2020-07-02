# Download and use an existing image which already has Conda installed and set up
FROM zeeros/tf2-env:latest
# Download the training dataset
RUN wget -q -P /usr/src/app/data --timeout=1 --tries=0 --retry-connrefused  http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-01-12.zip
