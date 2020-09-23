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

CREATE OR REPLACE PROCEDURE TrainLTV(
  MODEL_NAME STRING,
  DATA_TABLE STRING)

BEGIN

#[START TRAIN_MODEL]
EXECUTE IMMEDIATE """
CREATE OR REPLACE MODEL """ || MODEL_NAME || """
  OPTIONS(MODEL_TYPE='AUTOML_REGRESSOR',
          INPUT_LABEL_COLS=['target_monetary'],
          OPTIMIZATION_OBJECTIVE='MINIMIZE_MAE')
AS 
SELECT 
  * EXCEPT(customer_id)
FROM """ || DATA_TABLE;
#[END TRAIN_MODEL]

END