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


import os
import numpy as np
import apache_beam as beam

EMBEDDING_FILE_PREFIX = 'embeddings'

def get_query(dataset_name, table_name):
  query = f'''
    SELECT 
      item_Id,
      axis,
      factor_weights
    FROM 
      `{dataset_name}.{table_name}`
  '''
  return query


def parse_embeddings(bq_record):
  item_Id = bq_record['item_Id']
  axis = bq_record['axis']
  factor_weights = bq_record['factor_weights']
  dimensions = len(factor_weights)
  embedding = [0.0] * dimensions
  for idx, entry in enumerate(factor_weights):
    factor, weight = entry['factor'], entry['weight']
    embedding[int(factor) - 1] = float(weight)

  return (item_Id, embedding)


def average_embedding(entry):
  item_id, embedding_pair = entry
  embedding_pair = list(embedding_pair)
  
  if len(embedding_pair) == 2:
    embedding1, embedding2 = embedding_pair
    dimensions = len(embedding1)
    embedding = [0.0] * dimensions
    for idx in range(dimensions):
      embedding[idx] = (embedding1[idx] + embedding2[idx])
  else:
   embedding = embedding_pair[0]
  
  return item_id, embedding


def to_csv(entry):
  item_Id, embedding = entry
  csv_string = f'{item_Id},'
  csv_string += ','.join([str(value) for value in embedding])
  return csv_string

def run(bq_dataset_name, embeddings_table_name, output_dir, pipeline_args):

    pipeline_options = beam.options.pipeline_options.PipelineOptions(pipeline_args)
    project = pipeline_options.get_all_options()['project']
    with beam.Pipeline(options=pipeline_options) as pipeline:

      query = get_query(bq_dataset_name, embeddings_table_name)
      output_prefix = os.path.join(output_dir, EMBEDDING_FILE_PREFIX)
      
      _ = (
        pipeline
        | 'ReadFromBigQuery' >> beam.io.Read(beam.io.BigQuerySource(
            project=project, query=query, use_standard_sql=True, flatten_results=False))
        | 'ParseEmbeddings' >> beam.Map(parse_embeddings)
        | 'GroupByItem' >> beam.GroupByKey()
        | 'AverageItemEmbeddings' >> beam.Map(average_embedding)
        | 'ConvertToCsv' >> beam.Map(to_csv)
        | 'WriteToCloudStorage' >> beam.io.WriteToText(
            file_path_prefix = output_prefix,
            file_name_suffix = ".csv")
      )