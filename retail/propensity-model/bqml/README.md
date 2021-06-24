## License
```
Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

# How to build an end-to-end propensity to purchase solution using BigQuery ML and Kubeflow Pipelines
You’ll learn how to build a Propensity a model (if a Customer is going to buy) and how to orchestrate an ML Pipeline for doing so. You can use the code as a reference guide. Amend, replace, or add new AI Pipeline Components according to your use case. Please refer to the [Notebook](bqml_kfp_retail_propensity_to_purchase.ipynb), which contains further documentation and detailed instruction.

To see a step-by-step tutorial that walks you through implementing this solution, see [Predicting customer propensity to buy by using BigQuery ML and AI Platform](https://cloud.google.com/solutions/predicting-customer-propensity-to-buy).   

### The Notebook does the followings:
<ul>
    <li>Environment Setup</li>
    <ul>
        <li>Setup Cloud AI Platform Pipelines (using the CloudConsole)</li>
        <li>Install KFP client</li>
        <li>Install Python packages for Google Cloud Services</li>
    </ul>
    <li>Kubeflow Pipelines (KFP) Setup</li>
    <ul>
        <li>Prepare Data for the training</li>
        <ul>
            <li>Create/Validate a Google Cloud Storage Bucket/Folder</li>
            <li>Create the input table in BigQuery</li>
        </ul>
        <li>Train the model</li>
        <li>Evaluate the model</li>
        <li>Prepare the Model for batch prediction</li>
        <ul>
            <li>Prepare a test dataset (a table)</li>
            <li>Predict the model in BigQuery</li>
        </ul>
        <li>Prepare the Model for online prediction</li>
        <ul>
            <li>Create a new revision <i>(Model revision management)</i></li>
            <li>Export the BigQuery Model</li>
            <ul>
                <li>Export the Model from BigQuery to Google Cloud Storage</li>
                <li>Export the Training Stats to Google Cloud Storage</li>
                <li>Export the Eval Metrics to Google Cloud Storage</li>
            </ul>
            <li>Deploy to Cloud AI Platform Prediction</li>
            <li>Predict the model in Cloud AI Platform Prediction</li>
        </ul>
    </ul>
    <li>Data Exploration using BigQuery, Pandas, matplotlib</li>
    <li>SDLC methodologies Adherence (opinionated)</li>
        <ul>
            <li>Variables naming conventions</li>
            <ul>
                <li>Upper case Names for immutable variables</li>
                <li>Lower case Names for mutable variables</li>
                <li>Naming prefixes with <i>rpm_</i> or <i>RPM_</i></li>
            </ul>
            <li>Unit Tests</li>
            <li>Cleanup/Reset utility functions</li>
        </ul>
    <li>KFP knowledge share (demonstration)</li>
        <ul>
            <li>Pass inputs params through function args</li>
            <li>Pass params through pipeline args</li>
            <li>Pass Output from one Component as input of another</li>
            <li>Create an external Shared Volume available to all the Comp</li>
            <li>Use built in Operators</li>
            <li>Built light weight Component</li>
            <li>Set Component not to cache</li>
        </ul>
</ul>

### Architecture of the pipeline
![Pipeline components](images/MLOPs-Pipeline-Architecture.png?raw=true "Architecture of the Pipeline")

### Data Exploration and Visualization in the notebook

![Data Exploration](images/DataExploration.png?raw=true "Data Exploration")

![Data Visualization](images/DataVisualization.png?raw=true "Data Visualization")

## Running the Unit tests

Create a <b>local context</b> and use it to unit test the KFP Pipeline component locally. Below is an example of testing your component locally:
```python
# test locally create_gcs_bucket_folder
local_context = get_local_context()
import json
update_local_context (create_gcs_bucket_folder(
    json.dumps(local_context),
    local_context['RPM_GCP_STORAGE_BUCKET'],
    local_context['RPM_GCP_PROJECT'],
    local_context['RPM_DEFAULT_BUCKET_EXT'],
    local_context['RPM_GCP_STORAGE_BUCKET_FOLDER'],
    local_context['RPM_DEFAULT_BUCKET_FOLDER_NAME']
))
```

### Utility functions

Below is an utility function which purges GCS artifacts while unit/integration testing:

```python
#delete BQ Table if not needed...!!!BE CAREFUL!!!
def delete_table(table_id):
    from google.cloud import bigquery
    # Construct a BigQuery client object.
    client = bigquery.Client()
    # client.delete_table(table_id, not_found_ok=True)  # Make an API request.
    client.delete_table(table_id)  # Make an API request.
    print("Deleted table '{}'.".format(table_id))
#delete the table in the bigquery
delete_table(get_local_context()['rpm_table_id'])
```

## Mandatory Variables

You must set values of these parameters, please refer to the instructions in the Notebook for details:
```python
RPM_GCP_PROJECT:'<Your GCP Project>'
RPM_GCP_KFP_HOST='<Your KFP pipeline host>'
RPM_GCP_APPLICATION_CREDENTIALS="<Full path with the file name to the above downloaded json file>"
```

## A screen grab of the Output of the KFP pipeline
![KFP Graph](images/KFP-Graph.png?raw=true "KFP Graph")

## KFP Knowledge Share

The below code snippets demonstrates various KFP syntaxes. It shows various ways to pass parameters. You could use whatever works for you.

### Pass inputs params through function args example:
```python
# create BQ DS only if it doesn't exist
from typing import NamedTuple
def create_bq_ds (ctx:str, 
    RPM_GCP_PROJECT: str,
    RPM_BQ_DATASET_NAME: str, 
    RPM_LOCATION: str
    ) -> NamedTuple('Outputs', [
    ('rpm_context', str), 
    ('rpm_bq_ds_name', str), 
    ]):
```
![Input Output](images/KFP-Function_Params.png?raw=true "Input Output")

### Pass params through pipeline args example:
```python
def bq_googlestr_dataset_to_bq_to_caip_pipeline(
    data_path = all_vars['RPM_PVC_NAME'] #you can pass input variables
):
```

### Pass Output from one Kubeflow Pipelines Component as Input of another Kubeflow Pipelines Component example:
Output 'rpm_table_id' from load_bq_ds_op component passed as input to create_bq_ml_op comp
```python
    create_bq_ml_op = create_kfp_comp(create_bq_ml)(
        load_bq_ds_op.outputs['rpm_context'],
        all_vars['RPM_GCP_PROJECT'],
        all_vars['RPM_MODEL_NAME'],
        all_vars['RPM_DEFAULT_MODEL_NAME'],
        create_bq_ds_op.outputs['rpm_bq_ds_name'],
        load_bq_ds_op.outputs['rpm_table_id']
        )
```

### Create an external Shared Volume available to all the Kubeflow Pipelines Component example:
``` python
    #create a volume where the dataset will be temporarily stored.
    pvc_op = VolumeOp(
        name=all_vars['RPM_PVC_NAME'],
        resource_name=all_vars['RPM_PVC_NAME'],
        size="20Gi",
        modes=dsl.VOLUME_MODE_RWO
    )
```

### Use built in Ops example:
``` python
    #create a volume where the dataset will be temporarily stored.
    pvc_op = VolumeOp(
        name=all_vars['RPM_PVC_NAME'],
        resource_name=all_vars['RPM_PVC_NAME'],
        size="20Gi",
        modes=dsl.VOLUME_MODE_RWO
    )
```

### Built light weight Kubeflow Pipelines Component example:
``` python
# converting functions to container operations
import kfp.components as comp
def create_kfp_comp(rpm_comp):
    return comp.func_to_container_op(
        func=rpm_comp, 
        base_image="google/cloud-sdk:latest")
```

### Set Kubeflow Pipelines Component not to cache example:
```python
    get_versioned_bqml_model_export_path_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
```

## Questions? Feedback?
If you have any questions or feedback, please open up a [new issue](https://github.com/GoogleCloudPlatform/analytics-componentized-patterns/issues).