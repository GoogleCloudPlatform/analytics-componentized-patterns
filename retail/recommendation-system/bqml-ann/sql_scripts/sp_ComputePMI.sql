CREATE OR REPLACE PROCEDURE @DATASET_NAME.sp_ComputePMI(
  IN min_item_frequency INT64,
  IN max_group_size INT64
)

BEGIN

  DECLARE total INT64;

  # Get items with minimum frequency
  CREATE OR REPLACE TABLE @DATASET_NAME.valid_item_groups
  AS

  # Create valid item set
  WITH 
  valid_items AS (
    SELECT item_Id, COUNT(group_Id) AS item_frequency
    FROM @DATASET_NAME.vw_item_groups
    GROUP BY item_Id
    HAVING item_frequency >= min_item_frequency
  ),

  # Create valid group set
  valid_groups AS (
    SELECT group_Id, COUNT(item_Id) AS group_size
    FROM @DATASET_NAME.vw_item_groups
    WHERE item_Id IN (SELECT item_Id FROM valid_items)
    GROUP BY group_Id
    HAVING group_size BETWEEN 2 AND max_group_size
  )

  SELECT item_Id, group_Id
  FROM @DATASET_NAME.vw_item_groups
  WHERE item_Id IN (SELECT item_Id FROM valid_items)
  AND group_Id IN (SELECT group_Id FROM valid_groups);

  # Compute pairwise cooc
  CREATE OR REPLACE TABLE @DATASET_NAME.item_cooc
  AS
  SELECT item1_Id, item2_Id, SUM(cooc) AS cooc
  FROM
  (
    SELECT
      a.item_Id item1_Id,
      b.item_Id item2_Id,
      1 as cooc
    FROM @DATASET_NAME.valid_item_groups a
    JOIN @DATASET_NAME.valid_item_groups b
    ON a.group_Id = b.group_Id
    AND a.item_Id < b.item_Id
  )
  GROUP BY  item1_Id, item2_Id;

  ###################################
  
  # Compute item frequencies
  CREATE OR REPLACE TABLE @DATASET_NAME.item_frequency
  AS
  SELECT item_Id, COUNT(group_Id) AS frequency
  FROM @DATASET_NAME.valid_item_groups
  GROUP BY item_Id;

  ###################################
  
  # Compute total frequency |D|
  SET total = (
    SELECT SUM(frequency)  AS total
    FROM @DATASET_NAME.item_frequency
  );
  
  ###################################
  
  # Add mirror item-pair cooc and same item frequency as cooc
  CREATE OR REPLACE TABLE @DATASET_NAME.item_cooc
  AS
  SELECT item1_Id, item2_Id, cooc
  FROM @DATASET_NAME.item_cooc
  UNION ALL
  SELECT item2_Id as item1_Id, item1_Id AS item2_Id, cooc
  FROM @DATASET_NAME.item_cooc
  UNION ALL
  SELECT item_Id as item1_Id, item_Id AS item2_Id, frequency as cooc
  FROM @DATASET_NAME.item_frequency;

  ###################################
  
  # Compute PMI
  CREATE OR REPLACE TABLE @DATASET_NAME.item_cooc
  AS
  SELECT
    a.item1_Id,
    a.item2_Id,
    a.cooc,
    LOG(a.cooc, 2) - LOG(b.frequency, 2) - LOG(c.frequency, 2) + LOG(total, 2) AS pmi
  FROM @DATASET_NAME.item_cooc a
  JOIN @DATASET_NAME.item_frequency b
  ON a.item1_Id = b.item_Id
  JOIN @DATASET_NAME.item_frequency c
  ON a.item2_Id = c.item_Id; 
END