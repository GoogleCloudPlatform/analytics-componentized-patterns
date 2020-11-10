FROM tensorflow/tfx:0.23.0

RUN pip install scann==1.1.1 google-cloud-bigquery==2.3.1

WORKDIR /pipeline
COPY *.py/ ./
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"