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
"""Census training pipeline DSL."""

import os
import sys
from typing import Dict, List, Text, Optional
from kfp import gcp
import tfx
from tfx.orchestration import data_types
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2
from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen

import config
import bq_components

EMBEDDING_LOOKUP_MODEL_NAME = 'embeddings_lookup'
SCANN_INDEX_MODEL_NAME = 'embeddings_scann'
LOOKUP_EXPORTER_MODULE='lookup_exporter.py'
SCANN_INDEXER_MODULE='scann_indexer.py'


def create_pipeline(pipeline_name: Text, 
                    pipeline_root: Text, 
                    bq_dataset_name: Text,
                    min_item_frequency: data_types.RuntimeParameter,
                    max_group_size: data_types.RuntimeParameter,
                    dimensions: data_types.RuntimeParameter,
                    num_leaves: data_types.RuntimeParameter,
                    ai_platform_training_args: Dict[Text, Text],
                    beam_pipeline_args: List[Text],
                    model_regisrty_uri: Text,
                    enable_cache: Optional[bool] = False) -> pipeline.Pipeline:
  """Implements the online news pipeline with TFX."""

  # Compute the PMI.
  pmi_computer = bq_components.compute_pmi(
    project_id=config.PROJECT_ID,
    dataset=bq_dataset_name,
    min_item_frequency=min_item_frequency,
    max_group_size=max_group_size
  )
  
  # Train the BQML Matrix Factorization model.
  bqml_trainer = bq_components.train_item_matching_model(
    project_id=config.PROJECT_ID,
    dataset=bq_dataset_name,
    item_cooc=pmi_computer.outputs.item_cooc,
    dimensions=dimensions,
  )
  
  # Extract the embeddings from the BQML model to a tabl.
  embeddings_extractor = bq_components.extract_embeddings(
    project_id=config.PROJECT_ID,
    dataset=bq_dataset_name,
    model=bqml_trainer.outputs.model
  )
  
  # Export embeddings from BigQuery to Cloud Storage.
  embeddings_exporter = BigQueryExampleGen(
    query=f'''
      SELECT item_Id, embedding
      FROM {bq_dataset_name}.item_embeddings
    ''',
    output_config=example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=1)])),
    instance_name='BQExportEmbeddings'
  )
  
  embeddings_exporter.add_upstream_node(embeddings_extractor)
  
  # Create an embedding lookup SavedModel.
  lookup_savedmodel_exporter = tfx.components.Trainer(
    custom_executor_spec=executor_spec.ExecutorClassSpec(
      trainer_executor.GenericExecutor),
    module_file=LOOKUP_EXPORTER_MODULE,
    train_args={'num_steps': 0},
    eval_args={'num_steps': 0},
    schema=tfx.types.Channel(tfx.types.standard_artifacts.Schema),
    examples=embeddings_exporter.outputs.examples,
    instance_name='ExportEmbeddingLookup'
  )

  # Push the embedding lookup model to model registry location.
  embedding_lookup_pusher = tfx.components.Pusher(
    model=lookup_savedmodel_exporter.outputs.model,
    push_destination=tfx.proto.pusher_pb2.PushDestination(
      filesystem=tfx.proto.pusher_pb2.PushDestination.Filesystem(
        base_directory=os.path.join(model_regisrty_uri, EMBEDDING_LOOKUP_MODEL_NAME))
    ),
    instance_name='EmbeddingLookupPusher'
  )
  
  # Build the ScaNN index.
  scann_indexer = tfx.components.Trainer(
    custom_executor_spec=executor_spec.ExecutorClassSpec(
      ai_platform_trainer_executor.GenericExecutor),
    module_file=SCANN_INDEXER_MODULE,
    train_args={'num_steps': num_leaves},
    eval_args={'num_steps': 0},
    schema = tfx.types.Channel(tfx.types.standard_artifacts.Schema),
    examples=embeddings_exporter.outputs.examples,
    custom_config={'ai_platform_training_args': ai_platform_training_args},
    instance_name='BuildScaNNIndex'
  )
  
  # Push the ScaNN index to model registry location.
  embedding_scann_pusher = tfx.components.Pusher(
    model=scann_indexer.outputs.model,
    push_destination=tfx.proto.pusher_pb2.PushDestination(
      filesystem=tfx.proto.pusher_pb2.PushDestination.Filesystem(
        base_directory=os.path.join(model_regisrty_uri, SCANN_INDEX_MODEL_NAME))
    ),
    instance_name='ScaNNIndexPusher'
  )
  
  components=[
    pmi_computer,
    bqml_trainer,
    embeddings_extractor,
    embeddings_exporter,
    lookup_savedmodel_exporter,
    embedding_lookup_pusher,
    scann_indexer,
    embedding_scann_pusher
  ]
  
  print('The pipeline consists of the following components:')
  print([component.id for component in components])
  
  return pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=pipeline_root,
    components=components,
  enable_cache=enable_cache,
  beam_pipeline_args=beam_pipeline_args)