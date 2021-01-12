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
"""Creates an ANN index."""

import logging

import google.auth
import numpy as np
import tfx
import tensorflow as tf

from google.cloud import bigquery
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import InputArtifact, OutputArtifact, Parameter
from tfx.types.experimental.simple_artifacts import Dataset 

from ann_service import IndexClient
from ann_types import ANNIndex

NUM_NEIGHBOURS = 10
MAX_LEAVES_TO_SEARCH = 200
METRIC = 'DOT_PRODUCT_DISTANCE'
FEATURE_NORM_TYPE = 'UNIT_L2_NORM'
CHILD_NODE_COUNT = 1000
APPROXIMATE_NEIGHBORS_COUNT = 50

@component
def create_index(
    project_id: Parameter[str],
    project_number: Parameter[str],
    region: Parameter[str],
    display_name: Parameter[str],
    dimensions: Parameter[int],
    item_embeddings: InputArtifact[Dataset],
    ann_index: OutputArtifact[ANNIndex]):
    
    index_client = IndexClient(project_id, project_number, region)
    
    logging.info('Creating index:')
    logging.info(f'    Index display name: {display_name}')
    logging.info(f'    Embeddings location: {item_embeddings.uri}')
    
    index_description = display_name
    index_metadata = {
        'contents_delta_uri': item_embeddings.uri,
        'config': {
            'dimensions': dimensions,
            'approximate_neighbors_count': APPROXIMATE_NEIGHBORS_COUNT,
            'distance_measure_type': METRIC,
            'feature_norm_type': FEATURE_NORM_TYPE,
            'tree_ah_config': {
                'child_node_count': CHILD_NODE_COUNT,
                'max_leaves_to_search': MAX_LEAVES_TO_SEARCH
            }
        }
    }
    
    operation_id = index_client.create_index(display_name, 
                                             index_description,
                                             index_metadata)
    response = index_client.wait_for_completion(operation_id, 'Waiting for ANN index', 45)
    index_name = response['name']
    
    logging.info('Index {} created.'.format(index_name))
  
    # Write the index name to metadata.
    ann_index.set_string_custom_property('index_name', 
                                         index_name)
    ann_index.set_string_custom_property('index_display_name', 
                                         display_name)
