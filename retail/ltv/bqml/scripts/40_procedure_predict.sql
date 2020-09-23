-- Copyright 2020 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     https://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

CREATE OR REPLACE PROCEDURE PredictLTV(
  MODEL_NAME STRING,
  TABLE_DATA STRING,
  PREDICT_FROM_DATE STRING,
  WINDOW_LENGTH INT64,
  TABLE_PREDICTION STRING)

BEGIN

-- Date at which an input transactions window starts.
DECLARE WINDOW_START STRING;

IF WINDOW_LENGTH != 0 THEN
  SET WINDOW_START = (SELECT CAST(DATE_SUB(PREDICT_FROM_DATE, INTERVAL WINDOW_LENGTH DAY) AS STRING));
ELSE
  SET WINDOW_START = "1900-01-01";
END IF;

IF PREDICT_FROM_DATE = 'NULL' THEN
  SET PREDICT_FROM_DATE = (SELECT CAST(CURRENT_DATE() AS STRING));
END IF;

EXECUTE IMMEDIATE """
CREATE OR REPLACE TABLE """ || TABLE_PREDICTION || """ AS
SELECT
  customer_id,
  monetary AS monetary_so_far,
  ROUND(predicted_target_monetary, 2) AS monetary_predicted,
  ROUND(predicted_target_monetary - monetary, 2) AS monetary_future
FROM
  ML.PREDICT(
    MODEL """ || MODEL_NAME || """,
    (
      SELECT
        customer_id,
        ROUND(monetary_orders, 2) AS monetary,
        cnt_orders AS frequency,
        recency,
        T,
        ROUND(recency/cnt_orders, 2) AS time_between,
        ROUND(avg_basket_value, 2) AS avg_basket_value,
        ROUND(avg_basket_size, 2) AS avg_basket_size,
        has_returns,
        CEIL(avg_time_to_return) AS avg_time_to_return,
        num_returns
      FROM (
        SELECT
          customer_id,
          SUM(value) AS monetary_orders,
          DATE_DIFF(MAX(order_day), MIN(order_day), DAY) AS recency,
          DATE_DIFF(PARSE_DATE('%Y-%m-%d', '""" || PREDICT_FROM_DATE || """'), MIN(order_day), DAY) AS T,
          COUNT(DISTINCT order_day) AS cnt_orders,
          AVG(qty_articles) avg_basket_size,
          AVG(value) avg_basket_value,
          CASE
            WHEN SUM(num_returns) > 0 THEN 'y'
            ELSE 'n'
          END AS has_returns,
          AVG(time_to_return) avg_time_to_return,
          SUM(num_returns) num_returns,
        FROM
          """ || TABLE_DATA || """
        WHERE
          order_day <= PARSE_DATE('%Y-%m-%d', '""" || PREDICT_FROM_DATE || """') AND
          order_day >= PARSE_DATE('%Y-%m-%d', '""" || WINDOW_START || """')
        GROUP BY
          customer_id
      )
    )
  )""";

END