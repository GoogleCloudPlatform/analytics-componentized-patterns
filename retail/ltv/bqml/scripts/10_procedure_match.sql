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
  FIELD_VISITOR_ID STRING,
  FIELD_ORDER_ID STRING,
  FILED_TRANSACTION_DATE STRING,
  FIELD_SKU STRING,
  FIELD_QTY STRING,
  FIELD_UNIT_PRICE STRING,
  SOURCE_TABLE STRING)

BEGIN

-- || because dynamic table names are not supported.
EXECUTE IMMEDIATE """
CREATE OR REPLACE TEMP TABLE Orders AS 
SELECT """ ||
  FIELD_VISITOR_ID || """ AS customer_id, """ ||
  FIELD_ORDER_ID || """ AS order_id, """ ||
  FILED_TRANSACTION_DATE || """ AS transaction_date, """ ||
  FIELD_SKU || """ AS product_sku, """ ||
  FIELD_QTY || """ AS qty, """ ||
  FIELD_UNIT_PRICE || """ AS unit_price 
FROM """ ||
  SOURCE_TABLE;

END