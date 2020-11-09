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


import tensorflow as tf
import numpy as np
import logging

VOCABULARY_FILE_NAME = 'vocabulary.txt'
class EmbeddingLookup(tf.keras.Model):

  def __init__(self, embedding_files_prefix, **kwargs):
    super(EmbeddingLookup, self).__init__(**kwargs)

    vocabulary = list()
    embeddings = list()

    # Read embeddings from tfrecord files.
    logging.info('Loading embeddings from files ...')
    embedding_files = tf.io.gfile.glob(embedding_files_prefix)
    dataset = tf.data.TFRecordDataset(embedding_files, compression_type="GZIP")
    for tfrecord in dataset:
      example = tf.train.Example.FromString(tfrecord.numpy())
      item_Id = example.features.feature['item_Id'].bytes_list.value[0].decode()
      embedding = np.array(example.features.feature['embedding'].float_list.value)
      vocabulary.append(item_Id)
      embeddings.append(embedding)
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

  logging.info('Instantiating embedding lookup model...')
  embedding_lookup_model = EmbeddingLookup(embedding_files_path)
  logging.info('Model is Instantiated.')
  
  signatures = {
    'serving_default': embedding_lookup_model.__call__.get_concrete_function(),
  }

  logging.info('Exporting embedding lookup model as a SavedModel...')
  tf.saved_model.save(embedding_lookup_model, model_output_dir, signatures=signatures)
  logging.info('SavedModel is exported.')