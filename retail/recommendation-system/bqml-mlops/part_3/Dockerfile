FROM python:3.7

# Install Scikit-Learn
# Scikit-learn 0.20 was the last version to support Python 2.7 and Python 3.4.
# Scikit-learn now requires Python 3.6 or newer.
RUN python -m pip install --no-cache -I scikit-learn==0.23.2

# Install pandas
RUN python -m pip install --no-cache -I pandas==1.0.5

# Install Google SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y