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
"""Exports embeddings from a BQ table to a GCS location."""

import logging

from google.cloud import bigquery

import tfx
import tensorflow as tf

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import InputArtifact, OutputArtifact, Parameter

from tfx.types.experimental.simple_artifacts import Dataset 

BQDataset = Dataset

@component
def export_embeddings(
    project_id: Parameter[str],
    gcs_location: Parameter[str],
    item_embeddings_bq: InputArtifact[BQDataset],
    item_embeddings_gcs: OutputArtifact[Dataset]):
    
    filename_pattern = 'embedding-*.json'
    gcs_location = gcs_location.rstrip('/')
    destination_uri = f'{gcs_location}/{filename_pattern}'
    
    _, table_name = item_embeddings_bq.get_string_custom_property('table_name').split(':')
  
    logging.info(f'Exporting item embeddings from: {table_name}')
  
    bq_dataset, table_id = table_name.split('.')
    client = bigquery.Client(project=project_id)
    dataset_ref = bigquery.DatasetReference(project_id, bq_dataset)
    table_ref = dataset_ref.table(table_id)
    job_config = bigquery.job.ExtractJobConfig()
    job_config.destination_format = bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON

    extract_job = client.extract_table(
        table_ref,
        destination_uris=destination_uri,
        job_config=job_config
    )  
    extract_job.result() # Wait for resuls
    
    logging.info(f'Embeddings export completed. Output in {gcs_location}')
  
    # Write the location of the embeddings to metadata.
    item_embeddings_gcs.uri = gcs_location

 
