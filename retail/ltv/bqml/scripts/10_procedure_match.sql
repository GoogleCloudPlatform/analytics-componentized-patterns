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
--
-- Transforms raw data to match fields of the template

CREATE OR REPLACE PROCEDURE MatchFields(
  SOURCE_TABLE STRING)

BEGIN

-- || because dynamic table names are not supported.
EXECUTE IMMEDIATE """
CREATE OR REPLACE TEMP TABLE Orders AS 
SELECT 
  CAST(customer_id AS STRING) AS customer_id,
  order_id AS order_id,
  transaction_date AS transaction_date,
  product_sku AS product_sku,
  qty AS qty,
  unit_price AS unit_price 
FROM """ ||
  SOURCE_TABLE;

END