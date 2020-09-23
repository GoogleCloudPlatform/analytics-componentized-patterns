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
-- Extracts top LTV customers.
CREATE OR REPLACE PROCEDURE ExtractTopEmails(
  TOP_LTV_RATIO FLOAT64,
  TABLE_SOURCE STRING,
  TABLE_CRM STRING,
  TABLE_OUTPUT STRING)

BEGIN

EXECUTE IMMEDIATE """
CREATE OR REPLACE TABLE """|| TABLE_OUTPUT || """ AS (
  SELECT
  p.customer_id,
  monetary_future,
  c.email AS email
FROM (
  SELECT
    customer_id,
    monetary_future,
    PERCENT_RANK() OVER (ORDER BY monetary_future DESC) AS percent_rank_monetary
  FROM
      """ || TABLE_SOURCE || """ ) p
  -- This creates fake emails. You need to join with your own CRM table.
  INNER JOIN (
  SELECT
    customer_id,
    email
  FROM
    """ || TABLE_CRM || """ ) c
ON
  p.customer_id = CAST(c.customer_id AS STRING)
WHERE
  -- Decides the size of your list of emails. For similar-audience use cases 
  -- where you need to find a minimum of matching emails, 20% should provide
  -- enough potential emails.
  percent_rank_monetary <= """ || TOP_LTV_RATIO || """
ORDER BY 
  monetary_future DESC
)""";

END