[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Analytics Componentized Patterns

From sample dataset to activation, these componentized patterns are designed to help you get the most out of [BigQuery ML](https://cloud.google.com/bigquery-ml/docs) and other Google Cloud products in production.

### Retail use cases
* Recommendation systems:
  * How to build a recommendation system on e-commerce data using BigQuery ML. ([Code][recomm_code] | [Blogpost][recomm_blog] | [Video](recomm_video))
  * How to build an item-item real-time recommendation system on song playlists data using BigQuery ML. ([Code][bqml_scann_code] | [Reference Guide][bqml_scann_guide])
* Propensity to purchase model:
  * How to build an end-to-end propensity to purchase solution using BigQuery ML and Kubeflow Pipelines. ([Code][propen_code] | [Blogpost][propen_blog])
* Activate on Lifetime Value predictions: 
  * How to predict the monetary value of your customers and extract emails of the top customers to use in Adwords for example to create similar audiences. Automation is done by a combination of BigQuery Scripting, Stored Procedure and bash script. ([Code][ltv_code])
* Clustering:
  * How to build customer segmentation through k-means clustering using BigQuery ML. ([Code][clustering_code] | [Blogpost](clustering_blog))
* Demand Forecasting:
  * How to build a time series demand forecasting model using BigQuery ML ([Code][demandforecasting_code] | [Blogpost](demandforecasting_blog) | [Video](demandforecasting_video))


### Gaming use cases
* Propensity to churn model:
  * Churn prediction for game developers using Google Analytics 4 (GA4) and BigQuery ML. ([Code][gaming_propen_code] | Blogpost(TBA)) | Video(TBA))

[gaming_propen_code]: gaming/propensity-model/bqml
[gaming_propen_blog]: https://medium.com/google-cloud/
[Video]: https://medium.com/google-cloud/
[recomm_code]: retail/recommendation-system/bqml
[recomm_blog]: https://medium.com/google-cloud/how-to-build-a-recommendation-system-on-e-commerce-data-using-bigquery-ml-df9af2b8c110
[recomm_video]: https://youtube.com/watch?v=sEx8RwvT_-8
[bqml_scann_code]: retail/recommendation-system/bqml-scann
[bqml_scann_guide]: https://cloud.google.com/solutions/real-time-item-matching
[propen_code]: retail/propensity-model/bqml
[propen_blog]: https://medium.com/google-cloud/how-to-build-an-end-to-end-propensity-to-purchase-solution-using-bigquery-ml-and-kubeflow-pipelines-cd4161f734d9
[ltv_code]: retail/ltv/bqml
[clustering_code]: retail/clustering/bqml
[clustering_blog]: https://towardsdatascience.com/how-to-build-audience-clusters-with-website-data-using-bigquery-ml-6b604c6a084c
[demandforecasting_code]: retail/time-series/bqml-demand-forecasting
[demandforecasting_blog]: https://cloud.google.com/blog/topics/developers-practitioners/how-build-demand-forecasting-models-bigquery-ml
[demandforecasting_video]: https://www.youtube.com/watch?v=dwOt68CevYA

## Disclaimer
This is not an officially supported Google product.

All files in this repository are under the [Apache License, Version 2.0](LICENSE.txt) unless noted otherwise.
