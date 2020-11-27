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
import apache_beam as beam

EMBEDDING_FILE_PREFIX = 'embeddings'

def get_query(dataset_name, table_name):
  query = f'''
    SELECT 
      item_Id,
      embedding
    FROM 
      `{dataset_name}.{table_name}`;
  '''
  return query


def to_csv(entry):
  item_Id = entry['item_Id']
  embedding = entry['embedding']
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
        | 'ReadFromBigQuery' >> beam.io.ReadFromBigQuery(
            project=project, query=query, use_standard_sql=True, flatten_results=False)
        | 'ConvertToCsv' >> beam.Map(to_csv)
        | 'WriteToCloudStorage' >> beam.io.WriteToText(
            file_path_prefix = output_prefix,
            file_name_suffix = ".csv")
      )