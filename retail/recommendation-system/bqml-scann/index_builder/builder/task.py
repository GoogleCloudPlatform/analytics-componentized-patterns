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
from . import indexer

def get_args():

  args_parser = argparse.ArgumentParser()

  args_parser.add_argument(
    '--embedding-files-path',
    help='GCS or local paths to embedding files',
    required=True
  )

  args_parser.add_argument(
    '--output-dir',
    help='GCS or local paths to output index file',
    required=True
  )

  args_parser.add_argument(
    '--num-leaves',
    help='Number of trees to build in the index',
    default=250,
    type=int
  )

  args_parser.add_argument(
    '--job-dir',
    help='GCS or local paths to job package'
  )

  return args_parser.parse_args()


def main():
  args = get_args()
  indexer.build(
    embedding_files_pattern=args.embedding_files_path, 
    output_dir=args.output_dir,
    num_leaves=args.num_leaves
  )
    
if __name__ == '__main__':
    main()