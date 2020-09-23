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

# Reads command line arguments.
while [[ $# -gt 0 ]]
do
  case ${1} in
  -p|--project-id)
      shift
      PROJECT_ID="${1}"
      ;;
  -d|--dataset-id)
      shift
      DATASET_ID="${1}"
      ;;
  -o|table-sales)
      shift
      TABLE_SALES="${1}"
      ;;
  -c|table-crm)
      shift
      TABLE_CRM="${1}"
      ;;
  -t|--train-model-id)
      shift
      TRAIN_MODEL_ID="${1}"
      ;;
  -u|--use-model-id)
      shift
      USE_MODEL_ID="${1}"
      ;;
  -w|--window-length)
      shift
      WINDOW_LENGTH="${1}"
      ;;
  -s|--window-step)
      shift
      WINDOW_STEP="${1}"
      ;;
  -i|--window-step-initial)
      shift
      WINDOW_STEP_INITIAL="${1}"
      ;;
  -l|--length-future)
      shift
      LENGTH_FUTURE="${1}"
      ;;
  -m|--max-stdv-monetary)
      shift
      MAX_STDV_MONETARY="${1}"
      ;;
  -q|--max-stdv-qty)
      shift
      MAX_STDV_QTY="${1}"
      ;;
  -r|--top-ltv-ratio)
      shift
      TOP_LTV_RATIO="${1}"
      ;;
  -h|--help)
      echo "FLAGS"
      echo -e "  --project-id=PROJECT_ID, -p"
      echo -e "     [Required] Project ID."
      echo -e "  --dataset-id=DATASET_ID, -d"
      echo -e "     [Required] Dataset ID."
      echo -e "  --train-model-id=TRAIN_MODEL_ID, -t"
      echo -e "     Name of the trained model. Set to *null* if you do not want to train a model."
      echo -e "  --use-model-id=USE_MODEL_ID, -u"
      echo -e "     Name of the model to use for predictions. Must include dataset: [DATASET].[MODEL]"
      echo -e "  --window-length=WINDOW_LENGTH, -w"
      echo -e "     Time range in days for input transactions."
      echo -e "  --window-step=WINDOW_STEP, -s"
      echo -e "     Time in days between two windows. Equivalent to the time between two threshold dates."
      echo -e "  --window-step-initial=WINDOW_STEP_INITIAL, -i"
      echo -e "     Initial time in days before setting the first threshold date."
      echo -e "  --length-future=LENGTH_FUTURE, -l"
      echo -e "     Time in days for which to make a prediction."
      echo -e "  --max-stdv-monetary=MAX_STDV_MONETARY, -m"
      echo -e "     Standard deviation of the monetary value per customer above which the script removes transactions."
      echo -e "  --max-stdv-qty=MAX_STDV_QTY, -q"
      echo -e "     Standard deviation of the quantity value per customer above which the script removes transactions."
      echo -e "  --top-ltv-ratio=TOP_LTV_RATIO, -r"
      echo -e "     Percentage of the top customers to extract."
      exit
      ;;
  *)
      echo "Flag not supported. See ./run.sh --help"
      exit 1
      shift
      ;;
  esac
  shift
done

# Sets default values if missing from the command line.
NOW=$(date +"%Y%m%d%H%M%S")
TRAIN_MODEL_ID="${TRAIN_MODEL_ID:-$DATASET_ID.model_$NOW}"
USE_MODEL_ID="${USE_MODEL_ID:-$DATASET_ID.model_$NOW}"
WINDOW_LENGTH=${WINDOW_LENGTH:-0}
WINDOW_STEP=${WINDOW_STEP:-30}
WINDOW_STEP_INITIAL=${WINDOW_STEP_INITIAL:-90}
LENGTH_FUTURE=${LENGTH_FUTURE:-30}
MAX_STDV_MONETARY=${MAX_STDV_MONETARY:-500}
MAX_STDV_QTY=${MAX_STDV_QTY:-100}
TOP_LTV_RATIO=${TOP_LTV_RATIO:-0.2}

SOURCE_TABLE="${TABLE_SALES:-$DATASET_ID.sales}"  # Original data dump.
TABLE_CRM="${TABLE_CRM:-$DATASET_ID.crm}"         # Table with user information.
TABLE_AGGRED="${DATASET_ID}.aggred"               # Used to create training dataset and to run predictions.
TABLE_ML="${DATASET_ID}.ml"                       # Used to train the model
TABLE_PREDICTIONS="${DATASET_ID}.predictions"     # Table where to output predictions.
TABLE_EMAILS="${DATASET_ID}.top_emails"           # Table where to output top emails.

# Project and Dataset IDs are required.
if [ -z "$PROJECT_ID" ]; then
  echo "Please specify a project id. See ./run.sh --help"
  exit 1
fi

if [ -z "$DATASET_ID" ]; then
  echo "Please specify a dataset id. See ./run.sh --help"
  exit 1
fi

echo "---------------------------------------"
echo "Runs script with the follow parameters."
echo "PROJECT_ID: ${PROJECT_ID}"
echo "DATASET_ID: ${DATASET_ID}"
echo "WINDOW_STEP: ${WINDOW_STEP}"
echo "WINDOW_STEP_INITIAL: ${WINDOW_STEP_INITIAL}"
echo "LENGTH_FUTURE: ${LENGTH_FUTURE}"
echo "MAX_STDV_MONETARY: ${MAX_STDV_MONETARY}"
echo "MAX_STDV_QTY: ${MAX_STDV_QTY}"
echo "TOP_LTV_RATIO: ${TOP_LTV_RATIO}"
echo "Source table for transactions is: ${SOURCE_TABLE}"
echo "Source table for CRM is: ${TABLE_CRM}"
echo "Will train a model named: ${TRAIN_MODEL_ID}"
echo "Will run predictions using model: ${USE_MODEL_ID}"
echo "--------------------------------------"

gcloud config set project $PROJECT_ID

bq show ${PROJECT_ID}:${DATASET_ID} || bq mk ${PROJECT_ID}:${DATASET_ID}

# Load example datasets from public GCS to BQ if they don't exist.
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

store_procedure 00_procedure_persist.sql
store_procedure 10_procedure_match.sql
store_procedure 20_procedure_prepare.sql

run_action """
  CALL MatchFields('${SOURCE_TABLE}');
  CALL PrepareForML(
    CAST('${MAX_STDV_MONETARY}' AS INT64),
    CAST('${MAX_STDV_QTY}' AS INT64),
    CAST('${WINDOW_LENGTH}' AS INT64),
    CAST('${WINDOW_STEP}' AS INT64),
    CAST('${WINDOW_STEP_INITIAL}' AS INT64),
    CAST('${LENGTH_FUTURE}' AS INT64),
    '${TABLE_AGGRED}',
    '${TABLE_ML}');"""

store_procedure 30_procedure_train.sql
if [[ $TRAIN_MODEL_ID != "null" ]] ;then
  run_action """
    CALL TrainLTV(
      '${TRAIN_MODEL_ID}',
      '${TABLE_ML}');"""
fi

store_procedure 40_procedure_predict.sql
if [[ $USE_MODEL_ID != "null" ]] ;then
  run_action """
    CALL PredictLTV(
      '${USE_MODEL_ID}',
      '${TABLE_AGGRED}',
      'NULL',
      CAST('${WINDOW_LENGTH}' AS INT64),
      '${TABLE_PREDICTIONS}');"""
fi

store_procedure 50_procedure_top.sql
run_action """
  DECLARE TOP_EMAILS ARRAY<STRING>;

  CALL ExtractTopEmails(
      CAST('${TOP_LTV_RATIO}' AS FLOAT64),
      '${TABLE_PREDICTIONS}',
      '${TABLE_CRM}',
      '${TABLE_EMAILS}');"""