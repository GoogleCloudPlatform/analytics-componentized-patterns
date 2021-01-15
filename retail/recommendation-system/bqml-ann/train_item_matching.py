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
"""BigQuery compute PMI component."""

import logging

from google.cloud import bigquery

import tfx
import tensorflow as tf

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import InputArtifact, OutputArtifact, Parameter

from tfx.types.experimental.simple_artifacts import Dataset as BQDataset
from tfx.types.standard_artifacts import Model as BQModel


@component
def train_item_matching_model(
    project_id: Parameter[str],
    bq_dataset: Parameter[str],
    dimensions: Parameter[int],
    item_cooc: InputArtifact[BQDataset],
    bq_model: OutputArtifact[BQModel]):
    
    item_cooc_table = item_cooc.get_string_custom_property('table_name')
    stored_proc = f'{bq_dataset}.sp_TrainItemMatchingModel'
    query = f'''
        DECLARE dimensions INT64 DEFAULT {dimensions};
        CALL {stored_proc}(dimensions);
    '''
    model_name = 'item_matching_model'
  
    logging.info(f'Using item co-occurrence table: item_cooc_table')
    logging.info(f'Starting training of the model...')
    
    #client = bigquery.Client(project=project_id)
    #query_job = client.query(query)
    #query_job.result()
  
    logging.info(f'Model training completed. Output in {bq_dataset}.{model_name}.')
  
    # Write the location of the model to metadata. 
    bq_model.set_string_custom_property('model_name',
                                         f'{project_id}:{bq_dataset}.{model_name}')
   
  
