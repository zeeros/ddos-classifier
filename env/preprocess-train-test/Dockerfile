# Download and use an existing image which already has Conda installed and set up
FROM zeeros/tf2-env:latest
# Download the training dataset
RUN wget -q -P /usr/src/app/data --timeout=1 --tries=0 --retry-connrefused  http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-03-11.zip
# Copy source code
WORKDIR /usr/src/app
COPY metadata.json ./data
COPY preprocess_train.py ./
COPY preprocess_test.py ./
