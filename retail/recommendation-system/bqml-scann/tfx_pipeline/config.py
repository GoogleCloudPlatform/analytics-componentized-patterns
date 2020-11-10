# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The pipeline configurations."""

import os


PIPELINE_NAME=os.getenv('PIPELINE_NAME', 'bqml_scann_embedding_matching')
EMBEDDING_LOOKUP_MODEL_NAME=os.getenv('EMBEDDING_LOOKUP_MODEL_NAME', 'embeddings_lookup')
SCANN_INDEX_MODEL_NAME=os.getenv('SCANN_INDEX_MODEL_NAME', 'embeddings_scann')
PROJECT_ID=os.getenv('PROJECT_ID', 'tfx-cloudml')
REGION=os.getenv('REGION', 'europe-west1')
BQ_DATASET_NAME=os.getenv('BQ_DATASET_NAME', 'recommendations')
ARTIFACT_STORE_URI=os.getenv('ARTIFACT_STORE_URI', 'gs://tfx-cloudml-artifacts')
RUNTIME_VERSION=os.getenv('RUNTIME_VERSION', '2.2')
PYTHON_VERSION=os.getenv('PYTHON_VERSION', '3.7')
USE_KFP_SA=os.getenv('USE_KFP_SA', 'False')
ML_IMAGE_URI=os.getenv('ML_IMAGE_URI', 'tensorflow/tfx:0.23.0')
BEAM_RUNNER=os.getenv('BEAM_RUNNER', 'DirectRunner')
MODEL_REGISTRY_URI=os.getenv('MODEL_REGISTRY_URI', 'gs://tfx-cloudml-artifacts/model_registry')