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
"""TFX pipeline DSL."""

import os
import sys
from typing import Dict, List, Text, Optional
from kfp import gcp
import tfx
from tfx.proto import example_gen_pb2, infra_validator_pb2
from tfx.orchestration import pipeline, data_types
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen
from ml_metadata.proto import metadata_store_pb2

try:
  from . import bq_components
  from . import scann_evaluator
except:
  import bq_components
  import scann_evaluator


EMBEDDING_LOOKUP_MODEL_NAME = 'embeddings_lookup'
SCANN_INDEX_MODEL_NAME = 'embeddings_scann'
LOOKUP_EXPORTER_MODULE = 'lookup_exporter.py'
SCANN_INDEXER_MODULE = 'scann_indexer.py'
SCHEMA_DIR = 'schema'


def create_pipeline(pipeline_name: Text, 
                    pipeline_root: Text,
                    project_id: Text,
                    bq_dataset_name: Text,
                    min_item_frequency: data_types.RuntimeParameter,
                    max_group_size: data_types.RuntimeParameter,
                    dimensions: data_types.RuntimeParameter,
                    num_leaves: data_types.RuntimeParameter,
                    eval_min_recall: data_types.RuntimeParameter,
                    eval_max_latency: data_types.RuntimeParameter,
                    ai_platform_training_args: Dict[Text, Text],
                    beam_pipeline_args: List[Text],
                    model_regisrty_uri: Text,
                    metadata_connection_config: Optional[
                      metadata_store_pb2.ConnectionConfig] = None,
                    enable_cache: Optional[bool] = False) -> pipeline.Pipeline:
  """Implements the online news pipeline with TFX."""

  
  local_executor_spec = executor_spec.ExecutorClassSpec(
    trainer_executor.GenericExecutor)

  caip_executor_spec = executor_spec.ExecutorClassSpec(
    ai_platform_trainer_executor.GenericExecutor)
  
  # Compute the PMI.
  pmi_computer = bq_components.compute_pmi(
    project_id=project_id,
    bq_dataset=bq_dataset_name,
    min_item_frequency=min_item_frequency,
    max_group_size=max_group_size
  )
  
  # Train the BQML Matrix Factorization model.
  bqml_trainer = bq_components.train_item_matching_model(
    project_id=project_id,
    bq_dataset=bq_dataset_name,
    item_cooc=pmi_computer.outputs.item_cooc,
    dimensions=dimensions,
  )
  
  # Extract the embeddings from the BQML model to a table.
  embeddings_extractor = bq_components.extract_embeddings(
    project_id=project_id,
    bq_dataset=bq_dataset_name,
    bq_model=bqml_trainer.outputs.bq_model
  )
  
  # Export embeddings from BigQuery to Cloud Storage.
  embeddings_exporter = BigQueryExampleGen(
    query=f'''
      SELECT item_Id, embedding, bias,
      FROM {bq_dataset_name}.item_embeddings
    ''',
    output_config=example_gen_pb2.Output(
      split_config=example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=1)])),
    instance_name='BQExportEmbeddings'
  )
  
  # Add dependency from embeddings_exporter to embeddings_extractor.
  embeddings_exporter.add_upstream_node(embeddings_extractor)
  
  # Import embeddings schema.
  schema_importer = tfx.components.ImporterNode(
    source_uri=SCHEMA_DIR,
    artifact_type=tfx.types.standard_artifacts.Schema,
    instance_name='ImportEmbeddingsSchemaImpor',
  )
  
  # Generate stats for the embeddings for validation.
  stats_generator = tfx.components.StatisticsGen(
    examples=embeddings_exporter.outputs.examples,
    instance_name='GenerateEmbeddingsStats',
  )

  # Validate the embeddings stats against the schema.
  stats_validator = tfx.components.ExampleValidator(
    statistics=stats_generator.outputs.statistics,
    schema=schema_importer.outputs.result,
    instance_name='ValidateEmbeddingsStats',
  )

  # Create an embedding lookup SavedModel.
  lookup_savedmodel_exporter = tfx.components.Trainer(
    custom_executor_spec=local_executor_spec,
    module_file=LOOKUP_EXPORTER_MODULE,
    train_args={'num_steps': 0},
    eval_args={'num_steps': 0},
    schema=schema_importer.outputs.result,
    examples=embeddings_exporter.outputs.examples,
    instance_name='ExportEmbeddingLookup'
  )
  
  # Add dependency from stats_validator to lookup_savedmodel_exporter.
  lookup_savedmodel_exporter.add_upstream_node(stats_validator)

  # Infra-validate the embedding lookup model.
  infra_validator = tfx.components.InfraValidator(
    model=lookup_savedmodel_exporter.outputs.model,
    serving_spec=infra_validator_pb2.ServingSpec(
      tensorflow_serving=infra_validator_pb2.TensorFlowServing(
        tags=['latest']),
      local_docker=infra_validator_pb2.LocalDockerConfig(),
    ),
    validation_spec=infra_validator_pb2.ValidationSpec(
      max_loading_time_seconds=60,
      num_tries=3,
    ),
    instance_name='InfraValidateEmbeddingLookup'
  )

  # Push the embedding lookup model to model registry location.
  embedding_lookup_pusher = tfx.components.Pusher(
    model=lookup_savedmodel_exporter.outputs.model,
    # There is a bug here: https://github.com/tensorflow/tfx/blob/e1669693ef8739323a357d01a7a4801abf052ce4/tfx/components/pusher/executor.py#L103
    # It should be  infra_blessing.uri instead of model_blessing.uri
    # Thus setting iinfra_blessing and not model_blessing causes an error.
    # infra_blessing=infra_validator.outputs.blessing,
    push_destination=tfx.proto.pusher_pb2.PushDestination(
      filesystem=tfx.proto.pusher_pb2.PushDestination.Filesystem(
        base_directory=os.path.join(model_regisrty_uri, EMBEDDING_LOOKUP_MODEL_NAME))
    ),
    instance_name='PushEmbeddingLookup'
  )
  
  # Will be removed when the bug is fixed.
  embedding_lookup_pusher.add_upstream_node(infra_validator)
  
  # Build the ScaNN index.
  scann_indexer = tfx.components.Trainer(
    #custom_executor_spec=caip_executor_spec if ai_platform_training_args else local_executor_spec,
    custom_executor_spec=local_executor_spec,
    module_file=SCANN_INDEXER_MODULE,
    train_args={'num_steps': num_leaves},
    eval_args={'num_steps': 0},
    schema=schema_importer.outputs.result,
    examples=embeddings_exporter.outputs.examples,
    custom_config={'ai_platform_training_args': ai_platform_training_args},
    instance_name='BuildScaNNIndex'
  )
  
  # Evaluate and validate the ScaNN index.
  index_evaluator = scann_evaluator.IndexEvaluator(
    examples=embeddings_exporter.outputs.examples,
    schema=schema_importer.outputs.result,
    model=scann_indexer.outputs.model,
    min_recall=eval_min_recall,
    max_latency=eval_max_latency,
    instance_name='EvaluateScaNNIndex'
  )
  
  # Push the ScaNN index to model registry location.
  scann_index_pusher = tfx.components.Pusher(
    model=scann_indexer.outputs.model,
    model_blessing=index_evaluator.outputs.blessing,
    push_destination=tfx.proto.pusher_pb2.PushDestination(
      filesystem=tfx.proto.pusher_pb2.PushDestination.Filesystem(
        base_directory=os.path.join(model_regisrty_uri, SCANN_INDEX_MODEL_NAME))
    ),
    instance_name='PushScaNNIndex'
  )
  
  components=[
    pmi_computer,
    bqml_trainer,
    embeddings_extractor,
    embeddings_exporter,
    schema_importer,
    stats_generator,
    stats_validator,
    lookup_savedmodel_exporter,
    infra_validator,
    embedding_lookup_pusher,
    scann_indexer,
    index_evaluator,
    scann_index_pusher
  ]
  
  print('The pipeline consists of the following components:')
  print([component.id for component in components])
  
  return pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=pipeline_root,
    components=components,
    beam_pipeline_args=beam_pipeline_args,
    metadata_connection_config=metadata_connection_config,
    enable_cache=enable_cache
  )