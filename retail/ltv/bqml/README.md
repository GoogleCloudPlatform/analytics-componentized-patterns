# Activate on LTV predictions.
This guide refactors the [final part][series_final] of an existing series about predicting Lifetime Value (LTV). The series uses Tensorflow and shows multiple approaches such as statistical models or deep neural networks to predict the monetary value of customers. The final part leverages [AutoML Tables][automl_tables]. 

This document shows an opiniated way to predict the monetary value of your customers for a specific time in the future using historical data.

This updated version differs in the following:
- Predicts future monetary value for a specific period of time.
- Minimizes development time by using AutoML directly from [BigQuery ML][bq_ml].
- Uses two new datasets for sales and customer data.
- Creates additional training examples by moving the date that separates input and target orders (more details in the notebook)
- Shows how to activate the LTV predictions to create similar audiences in marketing tools.

The end to end flow assumes that you start with a data dump stored into BigQuery and runs through the following steps:

1. Match your dataset to the sales dataset template.
1. Create features from a list of orders.
1. Train a model using monetary value as a label.
1. Predict future monetary value of customers.
1. Extract the emails of the top customers.

For more general  information about LTV, read the [first part][series_first] of the series.

[series_final]:https://cloud.google.com/solutions/machine-learning/clv-prediction-with-automl-tables
[automl_tables]:https://cloud.google.com/automl-tables
[bq_ml]:https://cloud.google.com/bigquery-ml/docs/bigqueryml-intro
[series_first]:https://cloud.google.com/solutions/machine-learning/clv-prediction-with-offline-training-intro

## Files

There are two main sets of files:

**[1. ./notebooks](./notebooks)**

You can use the notebook in this folder to manually run the flow using example datasets. You can also used your own data.

**[2. ./scripts](./scripts)**

The scripts in this folder facilitate automation through BigQuery scripting, BigQuery stored procedures and bash scripting. Scripts use statements from the notebook to:
1. Transform data.
1. Train and use model to predict LTV.
1. Extract emails of the top LTV customers.

*Note: For production use cases, you can reuse the SQL statements from the scripts folder in pipeline tools such as Kubeflow Pipelines or Cloud Composer.*

The scripts assume that you already have the sales and crm datasets stored in BigQuery.

## Recommended flow

1. Do research in the Notebook.
1. Extract important SQL.
1. Write SQL scripts.
1. Test end-to-end flow through bash scripts.
1. Integrate into a data pipeline.
1. Run as part of a CI/CD pipeline.

This code shows you the steps 1 to 4.

## Run code

After you went through the notebook, you can run through all the steps at once using the [run.sh script][run_script].

1. If you use your own sales table, update the [matching query][matching_query] to transform your table into a table with a schema that the script understands.
1. Make sure that you can run the run.sh script

    ```chmod +x run.sh```

1. Check how to set parameters

    ```./run.sh --help```

1. Run the script
    ```./run.sh --project-id [YOUR_PROJECT_ID] --dataset-id [YOUR_DATASET_ID]


## Questions? Feedback?
If you have any questions or feedback, please open up a [new issue](https://github.com/GoogleCloudPlatform/analytics-componentized-patterns/issues).


## Disclaimer
This is not an officially supported Google product.

All files in this folder are under the Apache License, Version 2.0 unless noted otherwise.

[run_script]:./scripts/run.sh
[matching_query]:./scripts/10_procedure_match.sql