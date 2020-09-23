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

CREATE OR REPLACE PROCEDURE PrepareForML(
  MAX_STDV_MONETARY INT64,
  MAX_STDV_QTY INT64,
  WINDOW_LENGTH INT64,         -- How many days back for inputs transactions.
  WINDOW_STEP INT64,          -- How many days between thresholds.
  WINDOW_STEP_INITIAL INT64,  -- How many days for the first window.        
  LENGTH_FUTURE INT64,        -- How many days to predict for.  
  TABLE_FOR_PREDICTING STRING,
  TABLE_FOR_TRAINING STRING)

BEGIN

DECLARE MIN_DATE DATE;        -- Date of the first order in the dataset.                                     
DECLARE MAX_DATE DATE;        -- Date of the final order in the dataset.
DECLARE THRESHOLD_DATE DATE;  -- Date that separates inputs orders from target orders.  
DECLARE WINDOW_START DATE;    -- Date at which an input transactions window starts.
DECLARE STEP INT64 DEFAULT 1; -- Index of the window being run.

-- Aggregates per date per customers.
CREATE OR REPLACE TEMP TABLE Aggred AS
SELECT
  customer_id,
  order_day,
  ROUND(day_value_after_returns, 2) AS value,
  day_qty_after_returns as qty_articles,
  day_num_returns AS num_returns,
  CEIL(avg_time_to_return) AS time_to_return
FROM (
  SELECT
    customer_id,
    order_day,
    SUM(order_value_after_returns) AS day_value_after_returns,
    STDDEV(SUM(order_value_after_returns)) OVER(PARTITION BY customer_id ORDER BY SUM(order_value_after_returns)) AS stdv_value,
    SUM(order_qty_after_returns) AS day_qty_after_returns,
    STDDEV(SUM(order_qty_after_returns)) OVER(PARTITION BY customer_id ORDER BY SUM(order_qty_after_returns)) AS stdv_qty,
    CASE
      WHEN MIN(order_min_qty) < 0 THEN count(1)
      ELSE 0
    END AS day_num_returns,
    CASE
      WHEN MIN(order_min_qty) < 0 THEN AVG(time_to_return)
      ELSE NULL
    END AS avg_time_to_return
  FROM (
    SELECT 
      customer_id,
      order_id,
      -- Gives the order date vs return(s) dates.
      MIN(transaction_date) AS order_day,
      MAX(transaction_date) AS return_final_day,
      DATE_DIFF(MAX(transaction_date), MIN(transaction_date), DAY) AS time_to_return,
      -- Aggregates all products in the order 
      -- and all products returned later.
      SUM(qty * unit_price) AS order_value_after_returns,
      SUM(qty) AS order_qty_after_returns,
      -- If negative, order has qty return(s).
      MIN(qty) order_min_qty
    FROM 
      Orders
    GROUP BY
      customer_id,
      order_id)
  GROUP BY
    customer_id,
    order_day)
WHERE
  -- [Optional] Remove dates with outliers per a customer.
  (stdv_value < MAX_STDV_MONETARY
    OR stdv_value IS NULL) AND
  (stdv_qty < MAX_STDV_QTY
    OR stdv_qty IS NULL);

-- Creates the inputs and targets accross multiple threshold dates.
SET (MIN_DATE, MAX_DATE) = (
  SELECT AS STRUCT 
    MIN(order_day) AS min_days,
    MAX(order_day) AS max_days
  FROM
    Aggred
);

SET THRESHOLD_DATE = MIN_DATE;

CREATE OR REPLACE TEMP TABLE Featured
(
  -- dataset STRING,
  customer_id STRING,
  monetary FLOAT64,
  frequency INT64,
  recency INT64,
  T INT64,
  time_between FLOAT64,
  avg_basket_value FLOAT64,
  avg_basket_size FLOAT64,
  has_returns STRING,
  avg_time_to_return FLOAT64,
  num_returns INT64,
  -- threshold DATE,
  -- step INT64,
  target_monetary FLOAT64,
);

LOOP
  -- Can choose a longer original window in case 
  -- there were not many orders in the early days.
  IF STEP = 1 THEN
    SET THRESHOLD_DATE = DATE_ADD(THRESHOLD_DATE, INTERVAL WINDOW_STEP_INITIAL DAY); 
  ELSE
    SET THRESHOLD_DATE = DATE_ADD(THRESHOLD_DATE, INTERVAL WINDOW_STEP DAY);
  END IF;
  SET STEP = STEP + 1;

  IF THRESHOLD_DATE >= DATE_SUB(MAX_DATE, INTERVAL (WINDOW_STEP) DAY) THEN
    LEAVE;
  END IF;

  -- Takes all transactions before the threshold date unless you decide
  -- to use a different window lenght to test model performance.
  IF WINDOW_LENGTH != 0 THEN
    SET WINDOW_START = DATE_SUB(THRESHOLD_DATE, INTERVAL WINDOW_LENGTH DAY);
  ELSE
    SET WINDOW_START = MIN_DATE;
  END IF;

  INSERT Featured
  SELECT
    -- CASE
    --   WHEN THRESHOLD_DATE <= DATE_SUB(MAX_DATE, INTERVAL LENGTH_FUTURE DAY) THEN 'UNASSIGNED'
    --   ELSE 'TEST'
    -- END AS dataset,
    tf.customer_id,
    ROUND(tf.monetary_orders, 2) AS monetary,
    tf.cnt_orders AS frequency,
    tf.recency,
    tf.T,
    ROUND(tf.recency/cnt_orders, 2) AS time_between,
    ROUND(tf.avg_basket_value, 2) AS avg_basket_value,
    ROUND(tf.avg_basket_size, 2) AS avg_basket_size,
    has_returns,
    CEIL(avg_time_to_return) AS avg_time_to_return,
    num_returns,
    -- THRESHOLD_DATE AS threshold,
    -- STEP - 1 AS step,
    ROUND(tt.target_monetary, 2) AS target_monetary,
  FROM (
      -- This SELECT uses only data before THRESHOLD_DATE to make features.
      SELECT
        customer_id,
        SUM(value) AS monetary_orders,
        DATE_DIFF(MAX(order_day), MIN(order_day), DAY) AS recency,
        DATE_DIFF(THRESHOLD_DATE, MIN(order_day), DAY) AS T,
        COUNT(DISTINCT order_day) AS cnt_orders,
        AVG(qty_articles) avg_basket_size,
        AVG(value) avg_basket_value,
        CASE
          WHEN SUM(num_returns) > 0 THEN 'y'
          ELSE 'n'
        END AS has_returns,
        AVG(time_to_return) avg_time_to_return,
        THRESHOLD_DATE AS threshold,
        SUM(num_returns) num_returns,
      FROM
        Aggred
      WHERE
        order_day <= THRESHOLD_DATE AND
        order_day >= WINDOW_START
      GROUP BY
        customer_id
    ) tf
  INNER JOIN (
    -- This SELECT uses all orders that happened between threshold and 
    -- threshold + LENGTH_FUTURE to calculte the target monetary.
    SELECT
      customer_id,
      SUM(value) target_monetary
    FROM
      Aggred
    WHERE
      order_day <= DATE_ADD(THRESHOLD_DATE, INTERVAL LENGTH_FUTURE DAY)
      -- Overall value is similar to predicting only what's after threshold.
      -- and the prediction performs better. We can substract later.
      -- AND order_day > THRESHOLD_DATE
    GROUP BY
      customer_id) tt
  ON
    tf.customer_id = tt.customer_id;

END LOOP;

-- Persists the temporary ml table. Could do it directly from the above query
-- but this tutorial tries to limit String SQL statements as much as possible
-- and CREATE OR REPLACE TABLE without specifying a dataset is not supported.
-- The TrainLTV needs the table to be persisted.
EXECUTE IMMEDIATE """
CREATE OR REPLACE TABLE """|| TABLE_FOR_PREDICTING || """ AS (
SELECT * FROM Aggred )""";

EXECUTE IMMEDIATE """
CREATE OR REPLACE TABLE """|| TABLE_FOR_TRAINING || """ AS (
SELECT * FROM Featured )""";

-- CALL PersistData(TABLE_FOR_PREDICTING, "Aggred");
-- CALL PersistData(TABLE_FOR_TRAINING, "Featured");

END