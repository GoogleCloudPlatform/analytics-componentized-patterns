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


@component
def compute_pmi(
    project_id: Parameter[str],
    bq_dataset: Parameter[str],
    min_item_frequency: Parameter[int],
    max_group_size: Parameter[int],
    item_cooc: OutputArtifact[BQDataset]):
    
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
  
    #client = bigquery.Client(project=project_id)
    #query_job = client.query(query)
    #query_job.result() # Wait for the job to complete
  
    logging.info(f'Items PMI computation completed. Output in {bq_dataset}.{result_table}.')
  
    # Write the location of the output table to metadata.  
    item_cooc.set_string_custom_property('table_name',
                                         f'{project_id}:{bq_dataset}.{result_table}')
