FROM tensorflow/tfx:0.25.0

RUN pip install scann==1.1.1 google-cloud-bigquery==1.26.1 protobuf==3.13.0

WORKDIR /pipeline
COPY ./ ./
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"