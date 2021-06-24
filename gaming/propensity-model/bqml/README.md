## License
```
Copyright 2021 Google LLC

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

# Churn prediction for game developers using Google Analytics 4 (GA4) and BigQuery ML

This notebook showcases how you can use BigQuery ML to run propensity models on Google Analytics 4 data from your gaming app to determine the likelihood of specific users returning to your app.

Using this notebook, you'll learn how to:

- Explore the BigQuery export dataset for Google Analytics 4
- Prepare the training data using demographic and behavioural attributes
- Train propensity models using BigQuery ML
- Evaluate BigQuery ML models
- Make predictions using the BigQuery ML models
- Implement model insights in practical implementations

## Architecture Diagram

![Architecture Diagram](images/workflow.PNG?raw=true "Architecture Diagram")

## More resources

If you’d like to learn more about any of the topics covered in this notebook, check out these resources:

- [BigQuery export of Google Analytics data]
- [BigQuery ML quickstart]
- [Events automatically collected by Google Analytics 4]

## Questions? Feedback?
If you have any questions or feedback, please open up a [new issue](https://github.com/GoogleCloudPlatform/analytics-componentized-patterns/issues).

[BigQuery export of Google Analytics data]: https://support.google.com/analytics/answer/9358801
[BigQuery ML quickstart]: https://cloud.google.com/bigquery-ml/docs/bigqueryml-web-ui-start
[Events automatically collected by Google Analytics 4]: https://support.google.com/analytics/answer/9234069
