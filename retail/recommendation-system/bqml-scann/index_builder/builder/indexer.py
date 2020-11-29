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
import scann
import tensorflow as tf
import numpy as np
import math
import pickle

METRIC = 'dot_product'
DIMENSIONS_PER_BLOCK = 2
ANISOTROPIC_QUANTIZATION_THRESHOLD = 0.2
NUM_NEIGHBOURS = 10
NUM_LEAVES_TO_SEARCH = 200
REORDER_NUM_NEIGHBOURS = 200
TOKENS_FILE_NAME = 'tokens'


def load_embeddings(embedding_files_pattern):
    
  embedding_list = list()
  tokens = list()
  embed_files = tf.io.gfile.glob(embedding_files_pattern)
  print(f'{len(embed_files)} embedding files are found.')

  for file_idx, embed_file in enumerate(embed_files):
    print(f'Loading embeddings in file {file_idx+1} of {len(embed_files)}...')
    with tf.io.gfile.GFile(embed_file, 'r') as file_reader:
      lines = file_reader.readlines()
      for line in lines:
        parts = line.split(',')
        item_Id = parts[0]
        embedding = parts[1:]
        embedding = np.array([float(v) for v in embedding])
        normalized_embedding = embedding / np.linalg.norm(embedding)
        embedding_list.append(normalized_embedding)
        tokens.append(item_Id)
        
    print(f'{len(embedding_list)} embeddings are loaded.')
    
  return tokens, np.array(embedding_list)
    
    
def build_index(embeddings, num_leaves):
  
  data_size = embeddings.shape[0] 
  if not num_leaves:
    num_leaves = int(math.sqrt(data_size))
    
  print('Start building the ScaNN index...')
  scann_builder = scann.scann_ops.builder(embeddings, NUM_NEIGHBOURS, METRIC)
  scann_builder = scann_builder.tree(
    num_leaves=num_leaves, 
    num_leaves_to_search=NUM_LEAVES_TO_SEARCH, 
    training_sample_size=data_size)
  scann_builder = scann_builder.score_ah(
    DIMENSIONS_PER_BLOCK, 
    anisotropic_quantization_threshold=ANISOTROPIC_QUANTIZATION_THRESHOLD)
  scann_builder = scann_builder.reorder(REORDER_NUM_NEIGHBOURS)
  scann_index = scann_builder.build()
  print('ScaNN index is built.')
  
  return scann_index


def save_index(index, tokens, output_dir):
  print('Saving index as a SavedModel...')
  module = index.serialize_to_module()
  tf.saved_model.save(
    module, output_dir, signatures=None, options=None
  )
  print(f'Index is saved to {output_dir}')
  
  print(f'Saving tokens file...')
  tokens_file_path = os.path.join(output_dir, TOKENS_FILE_NAME)
  with tf.io.gfile.GFile(tokens_file_path, 'wb') as handle:
    pickle.dump(tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print(f'Item file is saved to {tokens_file_path}.')
 

def build(embedding_files_pattern, output_dir, num_leaves=None):
  print("Indexer started...")
  tokens, embeddings = load_embeddings(embedding_files_pattern)
  index = build_index(embeddings, num_leaves)
  save_index(index, tokens, output_dir)
  print("Indexer finished.")
    
    
    