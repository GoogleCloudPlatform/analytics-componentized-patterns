#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "Usage: ./run.sh <PROJECT> <DATASET> [<MODEL_TO_TRAIN>] [<MODEL_TO_USE>]"

PROJECT_ID=$1
DATASET_ID=$2

gcloud config set project $PROJECT_ID

NOW=$(date +"%Y%m%d%H%M%S")

SOURCE_TABLE="${DATASET_ID}.sales"            # Original data dump.
TABLE_CRM="${DATASET_ID}.crm"                 # Table with user information.
TABLE_AGGRED="${DATASET_ID}.aggred"           # Used to create training dataset and to run predictions.
TABLE_ML="${DATASET_ID}.ml"                   # Used to train the model
TABLE_PREDICTIONS="${DATASET_ID}.predictions" # Table where to output predictions.
TABLE_EMAILS="${DATASET_ID}.top_emails"       # Table where to output top emails.

TRAIN_MODEL_ID="${3:-$DATASET_ID.model_$NOW}"
USE_MODEL_ID="${4:-$DATASET_ID.model_$NOW}"

echo "Using project: ${PROJECT_ID}"
echo "Using dataset: ${DATASET_ID}"
echo "Will train a model named: ${TRAIN_MODEL_ID}"
echo "Will run predictions using model: ${USE_MODEL_ID}"

# Create data for example
bq show ${PROJECT_ID}:${DATASET_ID} || bq mk ${PROJECT_ID}:${DATASET_ID}

bq show ${TABLE_CRM} || \
bq load \
  --project_id $PROJECT_ID \
  --skip_leading_rows 1 \
  --max_bad_records 100000 \
  --replace \
  --field_delimiter "," \
  --autodetect \
  ${TABLE_CRM} \
  gs://solutions-public-assets/analytics-componentized-patterns/ltv/crm.csv

bq show ${SOURCE_TABLE} || \
bq load \
  --project_id $PROJECT_ID \
  --skip_leading_rows 1 \
  --max_bad_records 100000 \
  --replace \
  --field_delimiter "," \
  --autodetect \
  ${SOURCE_TABLE} \
  gs://solutions-public-assets/analytics-componentized-patterns/ltv/sales_*


function store_procedure() {
  echo ""
  echo "------------------------------------------------------"
  echo "--- Deploys procedure ${1}. ---"
  echo ""
  bq query \
  --project_id ${PROJECT_ID} \
  --dataset_id ${DATASET_ID} \
  --use_legacy_sql=false \
  < ${1}
}

function run_action() {
  echo ""
  echo "--------------------------------------------"
  echo " Run the following procedure:"
  echo "$@"
  echo ""
  bq query \
  --project_id ${PROJECT_ID} \
  --dataset_id ${DATASET_ID} \
  --use_legacy_sql=false \
  "$@"
}

#[START match_fields]
FIELD_VISITOR_ID="customer_id"
FIELD_ORDER_ID="order_id"
FILED_TRANSACTION_DATE="transaction_date"
FIELD_SKU="product_sku"
FIELD_QTY="qty"
FIELD_UNIT_PRICE="unit_price"
#[END match_fields]

# Pipeline parameters
WINDOW_STEP=30
WINDOW_STEP_INITIAL=90
LENGTH_FUTURE=30
MAX_STDV_MONETARY=500
MAX_STDV_QTY=100
TOP_LTV_RATIO=0.2

store_procedure 00_store_procedure_persist.sql
store_procedure 10_store_procedure_match.sql
store_procedure 20_store_procedure_prepare.sql

run_action """
  CALL MatchFields(
    '${FIELD_VISITOR_ID}',
    '${FIELD_ORDER_ID}',
    '${FILED_TRANSACTION_DATE}',
    '${FIELD_SKU}',
    '${FIELD_QTY}',
    '${FIELD_UNIT_PRICE}',
    '${SOURCE_TABLE}');

  CALL PrepareForML(
    CAST('${MAX_STDV_MONETARY}' AS INT64),
    CAST('${MAX_STDV_QTY}' AS INT64),
    CAST('${WINDOW_STEP}' AS INT64),
    CAST('${WINDOW_STEP_INITIAL}' AS INT64),
    CAST('${LENGTH_FUTURE}' AS INT64),
    '${TABLE_AGGRED}',
    '${TABLE_ML}');"""

store_procedure 30_store_procedure_train.sql
if [[ $TRAIN_MODEL_ID != "null" ]] ;then
  run_action """
    CALL TrainLTV(
      '${TRAIN_MODEL_ID}',
      '${TABLE_ML}');"""
fi

store_procedure 40_store_procedure_predict.sql
if [[ $USE_MODEL_ID != "null" ]] ;then
  run_action """
    CALL PredictLTV(
      '${USE_MODEL_ID}',
      '${TABLE_AGGRED}',
      'MAX(order_day)',
      '${TABLE_PREDICTIONS}');"""
fi

store_procedure 50_store_procedure_top.sql
run_action """
  DECLARE TOP_EMAILS ARRAY<STRING>;

  CALL ExtractTopEmails(
      CAST('${TOP_LTV_RATIO}' AS FLOAT64),
      '${TABLE_PREDICTIONS}',
      '${TABLE_EMAILS}');"""