# Activate on LTV predictions.
This folder shows an opiniated way to predict future monetary value of your customers using historical data.

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

## Disclaimer
This is not an officially supported Google product.

All files in this folder are under the Apache License, Version 2.0 unless noted otherwise.