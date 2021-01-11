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
"""Extracts embeddings to a BQ table."""

import logging

from google.cloud import bigquery

import tfx
import tensorflow as tf

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import InputArtifact, OutputArtifact, Parameter

from tfx.types.experimental.simple_artifacts import Dataset as BQDataset 
from tfx.types.standard_artifacts import Model as BQModel


@component
def extract_embeddings(
    project_id: Parameter[str],
    bq_dataset: Parameter[str],
    bq_model: InputArtifact[BQModel],
    item_embeddings: OutputArtifact[BQDataset]):
  
    embedding_model_name = bq_model.get_string_custom_property('model_name')
    stored_proc = f'{bq_dataset}.sp_ExractEmbeddings'
    query = f'''
        CALL {stored_proc}();
    '''
    embeddings_table = 'item_embeddings'

    logging.info(f'Extracting item embeddings from: {embedding_model_name}')
    
    #client = bigquery.Client(project=project_id)
    #query_job = client.query(query)
    #query_job.result() # Wait for the job to complete
  
    logging.info(f'Embeddings extraction completed. Output in {bq_dataset}.{embeddings_table}')
  
    # Write the location of the output table to metadata.
    item_embeddings.set_string_custom_property('table_name', 
                                                f'{project_id}:{bq_dataset}.{embeddings_table}')
    

 
