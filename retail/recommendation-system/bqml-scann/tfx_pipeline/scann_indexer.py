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
import sys
import scann
import tensorflow as tf
import numpy as np
import math
import pickle
import logging

METRIC = 'dot_product'
DIMENSIONS_PER_BLOCK = 2
ANISOTROPIC_QUANTIZATION_THRESHOLD = 0.2
NUM_NEIGHBOURS = 10
NUM_LEAVES_TO_SEARCH = 200
REORDER_NUM_NEIGHBOURS = 200
TOKENS_FILE_NAME = 'tokens'


def load_embeddings(embedding_files_pattern):
    
  embeddings = list()
  vocabulary = list()

  # Read embeddings from tfrecord files.
  logging.info('Loading embeddings from files ...')
  embedding_files = tf.io.gfile.glob(embedding_files_pattern)
  dataset = tf.data.TFRecordDataset(embedding_files, compression_type="GZIP")
  for tfrecord in dataset:
    example = tf.train.Example.FromString(tfrecord.numpy())
    item_Id = example.features.feature['item_Id'].bytes_list.value[0].decode()
    embedding = np.array(example.features.feature['embedding'].float_list.value)
    vocabulary.append(item_Id)
    embeddings.append(embedding)
  logging.info('Embeddings loaded.')
    
  return vocabulary, np.array(embeddings)
    
    
def build_index(embeddings, num_leaves):
  
  data_size = embeddings.shape[0] 
  if not num_leaves:
    num_leaves = int(math.sqrt(data_size))
    
  logging.info('Start building the ScaNN index...')
  scann_builder = scann.scann_ops.builder(embeddings, NUM_NEIGHBOURS, METRIC).tree(
    num_leaves=num_leaves, 
    num_leaves_to_search=NUM_LEAVES_TO_SEARCH, 
    training_sample_size=data_size).score_ah(
      DIMENSIONS_PER_BLOCK, anisotropic_quantization_threshold=ANISOTROPIC_QUANTIZATION_THRESHOLD).reorder(REORDER_NUM_NEIGHBOURS)
  scann_index = scann_builder.build()
  
  logging.info('ScaNN index is built.')
  
  return scann_index


def save_index(index, tokens, output_dir):
  logging.info('Saving index as a SavedModel...')
  module = index.serialize_to_module()
  tf.saved_model.save(
    module, output_dir, signatures=None, options=None
  )
  logging.info(f'Index is saved to {output_dir}')
  
  logging.info(f'Saving tokens file...')
  tokens_file_path = os.path.join(output_dir, TOKENS_FILE_NAME)
  with tf.io.gfile.GFile(tokens_file_path, 'wb') as handle:
    pickle.dump(tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)
  logging.info(f'Item file is saved to {tokens_file_path}.')


  
# TFX will call this function
def run_fn(params):
  embedding_files_path = params.train_files
  output_dir = params.serving_model_dir
  num_leaves = params.train_steps
  
  logging.info("Indexer started...")
  tokens, embeddings = load_embeddings(embedding_files_path)
  index = build_index(embeddings, num_leaves)
  save_index(index, tokens, output_dir)
  logging.info("Indexer finished.")
    
    
    