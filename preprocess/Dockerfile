# Download and use an existing image which already has Conda installed and set up
FROM continuumio/miniconda3:4.7.12
RUN apt-get update && apt-get install -y --no-install-recommends

# Because miniconda3 image releases may sometimes be behind we need to play catchup manually
RUN conda update conda && conda install "conda=4.8.3"

# Activate conda environment
COPY environment.yml .
ENV CONDA_ENV_NAME conda-env
RUN conda env create -n $CONDA_ENV_NAME -f environment.yml
RUN echo "source activate $CONDA_ENV_NAME" > /etc/bashrc
ENV PATH=/opt/conda/envs/$CONDA_ENV_NAME/bin:$PATH

COPY preprocess.py .
COPY call.py .
CMD python call.py
