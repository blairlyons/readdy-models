FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y python3-dev gcc build-essential

RUN mkdir working

COPY env.yml .
RUN conda env create -f env.yml
SHELL ["conda", "run", "-n", "models", "/bin/bash", "-c"]

COPY src/ .
COPY actin.sh /actin.sh
RUN chmod +x /actin.sh

ENTRYPOINT /actin.sh