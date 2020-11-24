# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BigQuery components."""

import os
import warnings
import logging

from google.cloud import bigquery

import tfx
import tensorflow as tf
from tfx.types.experimental.simple_artifacts import Dataset
from tfx.types.standard_artifacts import Artifact
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import InputArtifact, OutputArtifact, Parameter

from tfx.types.standard_artifacts import Model as BQModel


@component
def compute_pmi(
  project_id: Parameter[str],
  bq_dataset: Parameter[str],
  min_item_frequency: Parameter[int],
  max_group_size: Parameter[int],
  item_cooc: OutputArtifact[Dataset]):
  
  stored_proc = f'{bq_dataset}.sp_ComputePMI'
  query = f'''
      DECLARE min_item_frequency INT64;
      DECLARE max_group_size INT64;

      SET min_item_frequency = {min_item_frequency};
      SET max_group_size = {max_group_size};

      CALL {stored_proc}(min_item_frequency, max_group_size);
  '''
  result_table = 'item_cooc'

  logging.info(f'Starting computing PMI...')
  
  client = bigquery.Client(project=project_id)
  query_job = client.query(query)
  query_job.result() # Wait for the job to complete
  
  logging.info(f'Items PMI computation completed. Output in {bq_dataset}.{result_table}.')
  
  # Write the location of the output table to metadata.  
  item_cooc.set_string_custom_property('bq_dataset', bq_dataset)
  item_cooc.set_string_custom_property('bq_result_table', result_table)
    
    
@component
def train_item_matching_model(
  project_id: Parameter[str],
  bq_dataset: Parameter[str],
  dimensions: Parameter[int],
  item_cooc: InputArtifact[Dataset],
  bq_model: OutputArtifact[BQModel]):
    
  item_cooc_table = item_cooc.get_string_custom_property('bq_result_table')
  stored_proc = f'{bq_dataset}.sp_TrainItemMatchingModel'
  query = f'''
    DECLARE dimensions INT64 DEFAULT {dimensions};
    CALL {stored_proc}(dimensions);
  '''
  model_name = 'item_matching_model'
  
  logging.info(f'Using item co-occurrence table: {bq_dataset}.{item_cooc_table}')
  logging.info(f'Starting training of the model...')
    
  client = bigquery.Client(project=project_id)
  query_job = client.query(query)
  query_job.result()
  
  logging.info(f'Model training completed. Output in {bq_dataset}.{model_name}.')
  
  # Write the location of the model to metadata.  
  bq_model.set_string_custom_property('bq_dataset', bq_dataset)
  bq_model.set_string_custom_property('bq_model_name', model_name)
  
  
@component
def extract_embeddings(
  project_id: Parameter[str],
  bq_dataset: Parameter[str],
  bq_model: InputArtifact[BQModel],
  item_embeddings: OutputArtifact[Dataset]):
  
  embedding_model_name = bq_model.get_string_custom_property('bq_model_name')
  stored_proc = f'{bq_dataset}.sp_ExractEmbeddings'
  query = f'''
      CALL {stored_proc}();
  '''
  result_table = 'item_embeddings'

  logging.info(f'Extracting item embedding from: {bq_dataset}.{embedding_model_name}')
  logging.info(f'Starting exporting embeddings...')
  
  client = bigquery.Client(project=project_id)
  query_job = client.query(query)
  query_job.result() # Wait for the job to complete
  
  logging.info(f'Embeddings export completed. Output in {bq_dataset}.{result_table}')
  
  # Write the location of the output table to metadata.
  item_embeddings.set_string_custom_property('bq_dataset', bq_dataset)
  item_embeddings.set_string_custom_property('bq_result_table', result_table)