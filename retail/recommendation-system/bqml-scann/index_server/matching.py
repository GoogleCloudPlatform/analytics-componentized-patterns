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
import scann
import pickle
import os

TOKENS_FILE_NAME = 'tokens'


class ScaNNMatcher(object):

  def __init__(self, index_dir):
    print('Loading ScaNN index...')
    scann_module = tf.saved_model.load(index_dir)
    self.scann_index = scann.scann_ops.searcher_from_module(scann_module)
    tokens_file_path = os.path.join(index_dir, TOKENS_FILE_NAME)
    with tf.io.gfile.GFile(tokens_file_path, 'rb') as handle:
      self.tokens = pickle.load(handle)
    print('ScaNN index is loadded.')

  def match(self, vector, num_matches=10):
    embedding = np.array(vector)
    query = embedding / np.linalg.norm(embedding)
    matche_indices, _ = self.scann_index.search(query, final_num_neighbors=num_matches)
    match_tokens = [self.tokens[match_idx] for match_idx in matche_indices.numpy()]
    return match_tokens
