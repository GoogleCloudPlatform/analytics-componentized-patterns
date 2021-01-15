# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SQL Query Templates."""



CREATE_VW_ITEM_GROUPS = f"""
    CREATE OR REPLACE VIEW `{BQ_DATASET_NAME}.vw_item_groups`
    AS
    SELECT
      list_Id AS group_Id,
      track_Id AS item_Id
    FROM  
      `{BQ_DATASET_NAME}.{SOURCE_TABLE}` 
"""


COMPUTE_PMI = f"""
    DECLARE total INT64;

    # Get items with minimum frequency
    CREATE OR REPLACE TABLE {BQ_DATASET_NAME}.valid_item_groups
    AS

    # Create valid item set
    WITH 
    valid_items AS (
        SELECT item_Id, COUNT(group_Id) AS item_frequency
        FROM {BQ_DATASET_NAME}.vw_item_groups
        GROUP BY item_Id
        HAVING item_frequency >= {MIN_ITEM_FREQUENCY}
    ),

    # Create valid group set
    valid_groups AS (
        SELECT group_Id, COUNT(item_Id) AS group_size
        FROM {BQ_DATASET_NAME}.vw_item_groups
        WHERE item_Id IN (SELECT item_Id FROM valid_items)
        GROUP BY group_Id
        HAVING group_size BETWEEN 2 AND {MAX_GROUP_SIZE}
    )

    SELECT item_Id, group_Id
    FROM {BQ_DATASET_NAME}.vw_item_groups
    WHERE item_Id IN (SELECT item_Id FROM valid_items)
    AND group_Id IN (SELECT group_Id FROM valid_groups);

    # Compute pairwise cooc
    CREATE OR REPLACE TABLE {BQ_DATASET_NAME}.item_cooc
    AS
    SELECT item1_Id, item2_Id, SUM(cooc) AS cooc
    FROM
    (
        SELECT
        a.item_Id item1_Id,
        b.item_Id item2_Id,
        1 as cooc
        FROM {BQ_DATASET_NAME}.valid_item_groups a
        JOIN {BQ_DATASET_NAME}.valid_item_groups b
        ON a.group_Id = b.group_Id
        AND a.item_Id < b.item_Id
    )
    GROUP BY  item1_Id, item2_Id;

    ###################################
    
    # Compute item frequencies
    CREATE OR REPLACE TABLE {BQ_DATASET_NAME}.item_frequency
    AS
    SELECT item_Id, COUNT(group_Id) AS frequency
    FROM {BQ_DATASET_NAME}.valid_item_groups
    GROUP BY item_Id;

    ###################################
    
    # Compute total frequency |D|
    SET total = (
        SELECT SUM(frequency)  AS total
        FROM {BQ_DATASET_NAME}.item_frequency
    );
    
    ###################################
    
    # Add mirror item-pair cooc and same item frequency as cooc
    CREATE OR REPLACE TABLE {BQ_DATASET_NAME}.item_cooc
    AS
    SELECT item1_Id, item2_Id, cooc
    FROM {BQ_DATASET_NAME}.item_cooc
    UNION ALL
    SELECT item2_Id as item1_Id, item1_Id AS item2_Id, cooc
    FROM {BQ_DATASET_NAME}.item_cooc
    UNION ALL
    SELECT item_Id as item1_Id, item_Id AS item2_Id, frequency as cooc
    FROM {BQ_DATASET_NAME}.item_frequency;

    ###################################
    
    # Compute PMI
    CREATE OR REPLACE TABLE {BQ_DATASET_NAME}.item_cooc
    AS
    SELECT
        a.item1_Id,
        a.item2_Id,
        a.cooc,
        LOG(a.cooc, 2) - LOG(b.frequency, 2) - LOG(c.frequency, 2) + LOG(total, 2) AS pmi
    FROM {BQ_DATASET_NAME}.item_cooc a
    JOIN {BQ_DATASET_NAME}.item_frequency b
    ON a.item1_Id = b.item_Id
    JOIN {BQ_DATASET_NAME}.item_frequency c
    ON a.item2_Id = c.item_Id; 
    END
"""

TRAIN_MODEL = f"""
CREATE OR REPLACE MODEL {BQ_DATASET_NAME}.item_matching_model
    OPTIONS(
        MODEL_TYPE='matrix_factorization', 
        FEEDBACK_TYPE='implicit',
        WALS_ALPHA=1,
        NUM_FACTORS=({DIMENSIONS}),
        USER_COL='item1_Id', 
        ITEM_COL='item2_Id',
        RATING_COL='score',
        DATA_SPLIT_METHOD='no_split'
    )
    AS
    SELECT 
        item1_Id, 
        item2_Id, 
        cooc * pmi AS score
    FROM {BQ_DATASET_NAME}.item_cooc;
"""

EXPORT_EMBEDDINGS = f"""
CREATE OR REPLACE TABLE {BQ_DATASET_NAME}.item_embeddings AS
        WITH 
        step1 AS
        (
            SELECT 
                feature AS item_Id,
                factor_weights
            FROM
                ML.WEIGHTS(MODEL `{BQ_DATASET_NAME}..item_matching_model`)
            WHERE feature != 'global__INTERCEPT__'
        ),

        step2 AS
        (
            SELECT 
                item_Id, 
                factor, 
                SUM(weight) AS weight
            FROM step1,
            UNNEST(step1.factor_weights) AS embedding
            GROUP BY 
            item_Id,
            factor 
        )

        SELECT 
            item_Id as id, 
            ARRAY_AGG(weight ORDER BY factor ASC) embedding,
        FROM step2
        GROUP BY item_Id;

"""
