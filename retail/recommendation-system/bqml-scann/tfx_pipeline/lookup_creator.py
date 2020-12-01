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
"""Embedding lookup model."""


import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_transform.tf_metadata import schema_utils
import numpy as np
import logging

VOCABULARY_FILE_NAME = 'vocabulary.txt'
class EmbeddingLookup(tf.keras.Model):

  def __init__(self, embedding_files_prefix, schema_file_path, **kwargs):
    super(EmbeddingLookup, self).__init__(**kwargs)
    
    vocabulary = list()
    embeddings = list()

    logging.info('Loading schema...')
    schema = tfdv.load_schema_text(schema_file_path)
    feature_sepc = schema_utils.schema_as_feature_spec(schema).feature_spec
    logging.info('Schema is loadded.')
    
    def _gzip_reader_fn(filenames):
      return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    
    dataset = tf.data.experimental.make_batched_features_dataset(
      embedding_files_prefix, 
      batch_size=1, 
      num_epochs=1,
      features=feature_sepc,
      reader=_gzip_reader_fn,
      shuffle=False
    )

    # Read embeddings from tfrecord files.
    logging.info('Loading embeddings from files ...')
    for tfrecord_batch in dataset:
      vocabulary.append(tfrecord_batch["item_Id"].numpy()[0][0].decode())
      embeddings.append(tfrecord_batch["embedding"].numpy()[0])
    logging.info('Embeddings loaded.')
    
    embedding_size = len(embeddings[0])
    oov_embedding = np.zeros((1, embedding_size))
    self.embeddings = np.append(np.array(embeddings), oov_embedding, axis=0)
    logging.info(f'Embeddings: {self.embeddings.shape}')

    # Write vocabualry file.
    logging.info('Writing vocabulary to file ...')
    with open(VOCABULARY_FILE_NAME, 'w') as f:
      for item in vocabulary: 
        f.write(f'{item}\n')
    logging.info('Vocabulary file written and will be added as a model asset.')

    self.vocabulary_file = tf.saved_model.Asset(VOCABULARY_FILE_NAME)
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=vocabulary, values=list(range(len(vocabulary))))
    self.token_to_id = tf.lookup.StaticHashTable(
        initializer, default_value=len(vocabulary))

  @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
  def __call__(self, inputs):
    tokens = tf.strings.split(inputs, sep=None).to_sparse()
    ids = self.token_to_id.lookup(tokens) 
    embeddings = tf.nn.embedding_lookup_sparse(
        params=self.embeddings, 
        sp_ids=ids, 
        sp_weights=None, 
        combiner="mean"
    )
    return embeddings



# TFX will call this function
def run_fn(params):

  embedding_files_path = params.train_files
  model_output_dir = params.serving_model_dir
  schema_file_path = params.schema_file

  logging.info('Instantiating embedding lookup model...')
  embedding_lookup_model = EmbeddingLookup(embedding_files_path, schema_file_path)
  logging.info('Model is instantiated.')
  
  signatures = {
    'serving_default': embedding_lookup_model.__call__.get_concrete_function(),
  }

  logging.info('Exporting embedding lookup model as a SavedModel...')
  tf.saved_model.save(embedding_lookup_model, model_output_dir, signatures=signatures)
  logging.info('SavedModel is exported.')