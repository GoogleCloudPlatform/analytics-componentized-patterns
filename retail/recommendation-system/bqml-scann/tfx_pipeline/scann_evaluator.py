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
"""ScaNN index evaluator custom component."""

import os
import time
from typing import Any, Dict, List, Optional, Text, Union
import logging
import json

import tfx
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import artifact_utils
from tfx.utils import io_utils
from typing import Optional
from tfx import types

import tensorflow as tf
import numpy as np
import tensorflow_data_validation as tfdv
from tensorflow_transform.tf_metadata import schema_utils

try:
  from . import item_matcher
  from . import scann_indexer
except:
  import item_matcher
  import scann_indexer

  
QUERIES_SAMPLE_RATIO = 0.01
MAX_NUM_QUERIES = 10000
NUM_NEIGBHOURS = 20


class IndexEvaluatorSpec(tfx.types.ComponentSpec):
  
  INPUTS = {
    'examples': ChannelParameter(type=standard_artifacts.Examples),
    'schema': ChannelParameter(type=standard_artifacts.Schema),
    'model': ChannelParameter(type=standard_artifacts.Model),
  }
    
  OUTPUTS = {
    'evaluation': ChannelParameter(type=standard_artifacts.ModelEvaluation),
    'blessing': ChannelParameter(type=standard_artifacts.ModelBlessing),
  }
    
  PARAMETERS = {
    'min_recall': ExecutionParameter(type=float),
    'max_latency': ExecutionParameter(type=float),
  }


class ScaNNIndexEvaluatorExecutor(base_executor.BaseExecutor):
  
  def Do(self, 
         input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    
    if 'examples' not in input_dict:
      raise ValueError('Examples is missing from input dict.')
    if 'model' not in input_dict:
      raise ValueError('Model is missing from input dict.')
    if 'evaluation' not in output_dict:
      raise ValueError('Evaluation is missing from output dict.')
    if 'blessing' not in output_dict:
      raise ValueError('Blessing is missing from output dict.')
    
    valid = True
        
    self._log_startup(input_dict, output_dict, exec_properties)
    
    embedding_files_pattern = io_utils.all_files_pattern(
      artifact_utils.get_split_uri(input_dict['examples'], 'train'))
    
    schema_file_path = artifact_utils.get_single_instance(
      input_dict['schema']).uri + '/schema.pbtxt'
    
    vocabulary, embeddings = scann_indexer.load_embeddings(
      embedding_files_pattern, schema_file_path)
    
    num_embeddings = embeddings.shape[0]
    logging.info(f'{num_embeddings} embeddings are loaded.')
    num_queries = int(min(num_embeddings * QUERIES_SAMPLE_RATIO, MAX_NUM_QUERIES))
    logging.info(f'Sampling {num_queries} query embeddings for evaluation...')
    query_embedding_indices = np.random.choice(num_embeddings, num_queries)
    query_embeddings = np.take(embeddings, query_embedding_indices, axis=0)

    # Load Exact matcher
    exact_matcher = item_matcher.ExactMatcher(embeddings, vocabulary)
    exact_matches = []
    logging.info(f'Computing exact matches for the queries...')
    for query in query_embeddings:
      exact_matches.append(exact_matcher.match(query, NUM_NEIGBHOURS))
    logging.info(f'Exact matches are computed.')
    del num_embeddings, exact_matcher
    
    # Load ScaNN index matcher
    index_artifact = artifact_utils.get_single_instance(input_dict['model'])
    ann_matcher = item_matcher.ScaNNMatcher(index_artifact.uri + '/serving_model_dir')
    scann_matches = []
    logging.info(f'Computing ScaNN matches for the queries...')
    start_time = time.time()
    for query in query_embeddings:
      scann_matches.append(ann_matcher.match(query, NUM_NEIGBHOURS))
    end_time = time.time()
    logging.info(f'ScaNN matches are computed.')
    
    # Compute average latency
    elapsed_time = end_time - start_time
    current_latency = elapsed_time / num_queries

    # Compute recall
    current_recall = 0
    for exact, approx in zip(exact_matches, scann_matches):
      current_recall += len(set(exact).intersection(set(approx))) / NUM_NEIGBHOURS
    current_recall /= num_queries
    
    metrics = {
      'recall': current_recall,
      'latency': current_latency
    }
    
    min_recall = exec_properties['min_recall']
    max_latency = exec_properties['max_latency']
    
    logging.info(f'Average latency per query achieved {current_latency}. Maximum latency allowed: {max_latency}')
    logging.info(f'Recall acheived {current_recall}. Minimum recall allowed: {min_recall}')
    
    # Validate index latency and recall
    valid = (current_latency <= max_latency) and (current_recall >= min_recall)
    logging.info(f'Model is valid: {valid}')
    
    # Output the evaluation artifact.
    evaluation = artifact_utils.get_single_instance(output_dict['evaluation'])
    evaluation.set_string_custom_property('index_model_uri', index_artifact.uri)
    evaluation.set_int_custom_property('index_model_id', index_artifact.id)
    io_utils.write_string_file(
      os.path.join(evaluation.uri, 'metrics'), json.dumps(metrics))
    
    # Output the blessing artifact.
    blessing = artifact_utils.get_single_instance(output_dict['blessing'])
    blessing.set_string_custom_property('index_model_uri', index_artifact.uri)
    blessing.set_int_custom_property('index_model_id', index_artifact.id)

    if valid:
      io_utils.write_string_file(os.path.join(blessing.uri, 'BLESSED'), '')
      blessing.set_int_custom_property('blessed', 1)
    else:
      io_utils.write_string_file(os.path.join(blessing.uri, 'NOT_BLESSED'), '')
      blessing.set_int_custom_property('blessed', 0)


class IndexEvaluator(base_component.BaseComponent):
  
  SPEC_CLASS = IndexEvaluatorSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(ScaNNIndexEvaluatorExecutor)
    
  def __init__(self,
               examples: types.channel,
               schema: types.channel,
               model: types.channel,
               min_recall: float,
               max_latency: float,
               evaluation: Optional[types.Channel] = None,
               blessing: Optional[types.Channel] = None,
               instance_name=None):
    
    blessing = blessing or types.Channel(
      type=standard_artifacts.ModelBlessing,
      artifacts=[standard_artifacts.ModelBlessing()])
    
    evaluation = evaluation or types.Channel(
      type=standard_artifacts.ModelEvaluation,
      artifacts=[standard_artifacts.ModelEvaluation()])
    
    spec = IndexEvaluatorSpec(
      examples=examples, 
      schema=schema,
      model=model, 
      evaluation=evaluation,
      blessing=blessing, 
      min_recall=min_recall, 
      max_latency=max_latency
    )
        
    super().__init__(spec=spec, instance_name=instance_name)