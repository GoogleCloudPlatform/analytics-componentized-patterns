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

# How to build a time series demand forecasting model using BigQuery ML

The goal of [notebook](bqml_retail_demand_forecasting.ipynb) is to provide an end-to-end solution for forecasting the demand of multiple retail products. Using historical sales data of liquor products, you will learn how to train a demand forecasting model using BigQuery ML and how to visualize the forecasted values in a dashboard.

By the end of [this notebook](bqml_retail_demand_forecasting.ipynb), you will know how to:
* _pre-process data_ into the correct format needed to create a demand forecasting model with ARIMA using BigQuery ML
* _train the ARIMA model_ in BigQuery ML
* _evaluate the model_
* _make predictions on future demand using the model_
* _take action on the forecasted predictions:_
  * _create a dashboard to visualize the forecasted demand using Data Studio_


## Disclaimer
This is not an officially supported Google product.

All files in this folder are under the Apache License, Version 2.0 unless noted otherwise.

[run_script]:./scripts/run.sh
[matching_query]:./scripts/10_procedure_match.sql