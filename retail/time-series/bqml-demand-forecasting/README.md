# How to build a time series demand forecasting model using BigQuery ML

The goal of this repo is to provide an end-to-end solution for forecasting the demand of multiple retail products, using [this notebook](bqml_retail_demand_forecasting.ipynb) to walk through the steps. Learn how to use BigQuery ML to train a demand forecasting model on historical sales data of liquor products, and how to visualize the forecasted values in a dashboard. For an overview of the use case, see [Overview of a demand forecasting solution](https://cloud.google.com/architecture/demand-forecasting-overview).

After completing the notebook, you will know how to:

* Pre-process time series data into the correct format needed to create a demand forecasting model.
* Train the demand forecasting model in BigQuery ML.
* Evaluate the model.
* Make predictions about future demand using the model.
* Create a dashboard to visualize the forecasted demand using Data Studio.

This solution is intended for data engineers, data scientists, and data analysts
who build machine learning (ML) datasets and models to support business
decisions. It assumes that you have basic knowledge of the following:

* Machine learning concepts
* Python
* Standard SQL

## Dataset

This tutorial uses the public
[Iowa Liquor Sales](https://pantheon.corp.google.com/marketplace/product/iowa-department-of-commerce/iowa-liquor-sales)
dataset that is hosted on BigQuery. This dataset contains the
spirits purchase information of Iowa Class "E" liquor licensees from
January 1, 2012 until the present. For more information, see the
[official documentation by the State of Iowa](https://data.iowa.gov/Sales-Distribution/Iowa-Liquor-Sales/m3tr-qhgy).

## Costs 

This tutorial uses billable components of Google Cloud Platform (GCP):

* AI Platform
* BigQuery
* BigQuery ML

Learn about [BigQuery pricing](https://cloud.google.com/bigquery/pricing), [BigQuery ML
pricing](https://cloud.google.com/bigquery-ml/pricing) and use the [Pricing
Calculator](https://cloud.google.com/products/calculator/)
to generate a cost estimate based on your projected usage.

## Set up the GCP environment

1. [If you don't want to use an existing project, create a new GCP project.](https://console.cloud.google.com/cloud-resource-manager) When you first create an account, you get a $300 free credit towards your compute/storage costs.
1. [If you created a new project, make sure that billing is enabled for it.](https://cloud.google.com/billing/docs/how-to/modify-project)
1. [Enable the AI Platform, AI Platform Notebooks, and Compute Engine APIs in the project you plan to use.](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,notebooks.googleapis.com,compute_component)

## Run the notebook

1. Open the [bqml_retail_demand_forecasting.ipynb](bqml_retail_demand_forecasting.ipynb) notebook.
1. Click **Run on AI Platform Notebooks**.
2. In the GCP console, select the project you want to use to run this notebook.
3. If you have existing notebook instances in this project, select **Create a new notebook instance**.
4. For **Instance name**, type `demand-forecasting`.
5. Click **Create**.

## Clean up the GCP environment

Unless you plan to continue using the resources you created while using this notebook, you should delete them once you are done
to avoid incurring charges to your GCP account. You can either delete the project containing the resources, or
keep the project but delete just those resources.

Either way, you should remove the resources so you won't be billed for them in
the future. The following sections describe how to delete these resources.

### Delete the project

The easiest way to eliminate billing is to delete the project you created for
the solution.

1. In the Cloud Console, go to the [Manage resources page](https://pantheon.corp.google.com/cloud-resource-manager).
1. In the project list, select the project that you want to delete, and then click **Delete**.
1. In the dialog, type the project ID, and then click **Shut down** to delete the project.

### Delete the components

If you don't want to delete the project, delete the billable components of the solution.
These include:

1. The `demand-forecasting` AI Platform notebook instance.
2. The `bqmlforecast` BigQuery dataset.

## Disclaimer
This is not an officially supported Google product.

All files in this folder are under the Apache License, Version 2.0 unless noted otherwise.
