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
"""ScaNN index builder."""

import os
import sys
import scann
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_transform.tf_metadata import schema_utils
import numpy as np
import math
import pickle
import logging

METRIC = 'dot_product'
DIMENSIONS_PER_BLOCK = 2
ANISOTROPIC_QUANTIZATION_THRESHOLD = 0.2
NUM_NEIGHBOURS = 10
NUM_LEAVES_TO_SEARCH = 250
REORDER_NUM_NEIGHBOURS = 250
TOKENS_FILE_NAME = 'tokens'


def load_embeddings(embedding_files_pattern, schema_file_path):

  embeddings = list()
  vocabulary = list()
  
  logging.info('Loading schema...')
  schema = tfdv.load_schema_text(schema_file_path)
  feature_sepc = schema_utils.schema_as_feature_spec(schema).feature_spec
  logging.info('Schema is loaded.')

  def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    
  dataset = tf.data.experimental.make_batched_features_dataset(
    embedding_files_pattern, 
    batch_size=1, 
    num_epochs=1,
    features=feature_sepc,
    reader=_gzip_reader_fn,
    shuffle=False
  )

  # Read embeddings from tfrecord files.
  logging.info('Loading embeddings from files...')
  for tfrecord_batch in dataset:
    vocabulary.append(tfrecord_batch["item_Id"].numpy()[0][0].decode())
    embedding = tfrecord_batch["embedding"].numpy()[0]
    normalized_embedding = embedding / np.linalg.norm(embedding)
    embeddings.append(normalized_embedding)
  logging.info('Embeddings loaded.')
  embeddings = np.array(embeddings)
  
  return vocabulary, embeddings
    
    
def build_index(embeddings, num_leaves):
  
  data_size = embeddings.shape[0] 
  if not num_leaves:
    num_leaves = int(math.sqrt(data_size))
  logging.info(f'Indexing {data_size} embeddings with {num_leaves} leaves.')
    
  logging.info('Start building the ScaNN index...')
  scann_builder = scann.scann_ops.builder(embeddings, NUM_NEIGHBOURS, METRIC).tree(
    num_leaves=num_leaves, 
    num_leaves_to_search=NUM_LEAVES_TO_SEARCH, 
    training_sample_size=data_size).score_ah(
      DIMENSIONS_PER_BLOCK,
      anisotropic_quantization_threshold=ANISOTROPIC_QUANTIZATION_THRESHOLD).reorder(REORDER_NUM_NEIGHBOURS)
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
  schema_file_path = params.schema_file
  
  logging.info("Indexer started...")
  tokens, embeddings = load_embeddings(embedding_files_path, schema_file_path)
  index = build_index(embeddings, num_leaves)
  save_index(index, tokens, output_dir)
  logging.info("Indexer finished.")
    
    
    