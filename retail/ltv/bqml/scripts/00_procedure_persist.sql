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
-- Persists the data from a temporary table to a materialized table.
CREATE OR REPLACE PROCEDURE PersistData(
  TEMP_TABLE STRING,
  DEST_TABLE STRING)

BEGIN
  EXECUTE IMMEDIATE """
  CREATE OR REPLACE TABLE """|| DEST_TABLE || """ AS (
  SELECT * FROM """ || TEMP_TABLE || """)""";
END