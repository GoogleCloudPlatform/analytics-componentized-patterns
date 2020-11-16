# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KFP runner"""

import kfp
from kfp import gcp
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow import kubeflow_dag_runner

from typing import Optional, Dict, List, Text

import config
import pipeline

if __name__ == '__main__':

  # Set the values for the compile time parameters.
    
  ai_platform_training_args = {
    'project': config.PROJECT_ID,
    'region': config.REGION,
    'masterConfig': {
      'imageUri': config.ML_IMAGE_URI
    }
  }
   
  beam_pipeline_args = [
    f'--runner={config.BEAM_RUNNER}',
    '--experiments=shuffle_mode=auto',
    f'--project={config.PROJECT_ID}',
    f'--temp_location={config.ARTIFACT_STORE_URI}/beam/tmp',
    f'--region={config.REGION}',
  ]
    
  
  # Set the default values for the pipeline runtime parameters.
  
  min_item_frequency = data_types.RuntimeParameter(
      name='min-item-frequency',
      default=15,
      ptype=int
  )

  max_group_size = data_types.RuntimeParameter(
      name='max_group_size',
      default=100,
      ptype=int
  )
  
  dimensions = data_types.RuntimeParameter(
      name='dimensions',
      default=50,
      ptype=int
  )
  
  num_leaves = data_types.RuntimeParameter(
      name='num-leaves',
      default=0,
      ptype=int
  )
  
  eval_min_recall = data_types.RuntimeParameter(
      name='eval-min-recall',
      default=0.8,
      ptype=float
  )
  
  eval_max_latency = data_types.RuntimeParameter(
      name='eval-max-latency',
      default=0.01,
      ptype=float
  )
    
  pipeline_root = f'{config.ARTIFACT_STORE_URI}/{config.PIPELINE_NAME}/{kfp.dsl.RUN_ID_PLACEHOLDER}'

  # Set KubeflowDagRunner settings
  metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
    kubeflow_metadata_config = metadata_config,
    pipeline_operator_funcs = kubeflow_dag_runner.get_default_pipeline_operator_funcs(
      config.USE_KFP_SA == 'True'),
    tfx_image=config.ML_IMAGE_URI
  )

  # Compile the pipeline
  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
    pipeline.create_pipeline(
      pipeline_name=config.PIPELINE_NAME,
      pipeline_root=pipeline_root,
      project_id=config.PROJECT_ID,
      bq_dataset_name=config.BQ_DATASET_NAME,
      min_item_frequency=min_item_frequency,
      max_group_size=max_group_size,
      dimensions=dimensions,
      num_leaves=num_leaves,
      eval_min_recall=eval_min_recall,
      eval_max_latency=eval_max_latency,
      ai_platform_training_args=ai_platform_training_args,
      beam_pipeline_args=beam_pipeline_args,
      model_regisrty_uri=config.MODEL_REGISTRY_URI)
  )