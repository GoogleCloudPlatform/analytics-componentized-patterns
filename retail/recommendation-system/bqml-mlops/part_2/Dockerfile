FROM python:3.7

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

RUN pip3 install fire pyyaml pathlib
RUN pip3 install --upgrade kfp
RUN pip3 install google-cloud-bigquery


