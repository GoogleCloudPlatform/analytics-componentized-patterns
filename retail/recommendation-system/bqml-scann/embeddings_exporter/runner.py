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


import argparse
import pipeline

SETUP_FILE_PATH = './setup.py'


def get_args(argv):

  args_parser = argparse.ArgumentParser()

  args_parser.add_argument('--bq_dataset_name',
                           help='BigQuery dataset name.',
                           required=True)

  args_parser.add_argument('--embeddings_table_name',
                           help='BigQuery table name where the embeddings are stored.',
                           required=True)

  args_parser.add_argument('--output_dir',
                           help='GCS location where the embedding CSV files will be stored.',
                           required=True)

  return args_parser.parse_known_args()


def main(argv=None):
  args, pipeline_args = get_args(argv)
  pipeline_args.append('--setup_file={}'.format(SETUP_FILE_PATH))

  pipeline.run(
    args.bq_dataset_name, 
    args.embeddings_table_name, 
    args.output_dir, 
    pipeline_args)


if __name__ == '__main__':
  main()
