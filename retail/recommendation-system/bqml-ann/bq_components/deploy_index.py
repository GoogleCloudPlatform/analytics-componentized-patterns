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
"""Deploys an ANN index."""

import logging

import numpy as np
import tfx
import tensorflow as tf

from google.cloud import bigquery
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import InputArtifact, OutputArtifact, Parameter
from tfx.types.experimental.simple_artifacts import Dataset 

from ann_service import IndexDeploymentClient
from ann_types import ANNIndex
from ann_types import DeployedANNIndex


@component
def deploy_index(
    project_id: Parameter[str],
    project_number: Parameter[str],
    region: Parameter[str],
    vpc_name: Parameter[str],
    deployed_index_id: Parameter[str],
    ann_index: InputArtifact[ANNIndex],
    deployed_ann_index: OutputArtifact[DeployedANNIndex]
    ):
    
    deployment_client = IndexDeploymentClient(project_id, 
                                              project_number,
                                              region)
    
    index_name = ann_index.get_string_custom_property('index_name')
    index_display_name = ann_index.get_string_custom_property('index_display_name')
    endpoint_display_name = f'Endpoint for {index_display_name}'
    
    logging.info(f'Creating endpoint: {endpoint_display_name}')
    operation_id = deployment_client.create_endpoint(endpoint_display_name, vpc_name)
    response = deployment_client.wait_for_completion(operation_id, 'Waiting for endpoint', 30)
    endpoint_name = response['name']
    logging.info(f'Endpoint created: {endpoint_name}')
  
    #logging.info(f'Creating deployed index: {deployed_index_id}')
    #logging.info(f'                  from: {index_name}')
    #endpoint_id = endpoint_name.split('/')[-1]
    #index_id = index_name.split('/')[-1]
    #deployed_index_display_name = f'Deployed {index_display_name}'
    #operation_id = deployment_client.create_deployment(
    #    deployed_index_display_name, 
    #    deployed_index_id,
    #    endpoint_id,
    #    index_id)

    #response = deployment_client.wait_for_completion(operation_id, 'Waiting for deployment', 60)
    #logging.info('Index deployed!')
  
    #deployed_index_ip = deployment_client.get_deployment_grpc_ip(
    #    endpoint_id, deployed_index_id
    #)
    # Write the deployed index properties to metadata.
    deployed_ann_index.set_string_custom_property('endpoint_name', 
                                                  endpoint_name)
    #deployed_ann_index.set_string_custom_property('deployed_index_id', 
    #                                              deployed_index_id)
    #deployed_ann_index.set_string_custom_property('index_name', 
    #                                              index_name)
    #deployed_ann_index.set_string_custom_property('deployed_index_grpc_ip', 
    #                                              deployed_index_ip)
