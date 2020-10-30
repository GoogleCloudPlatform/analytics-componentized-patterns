CREATE OR REPLACE PROCEDURE recommendations.sp_ComputePMI(
  IN min_item_frequency INT64,
  IN max_group_size INT64,
  IN negative_sample_size INT64
)

BEGIN

  DECLARE total INT64;

  # Get items with minimum frequency
  CREATE OR REPLACE TABLE recommendations.valid_item_groups
  AS

  # Create valid item set
  WITH 
  valid_items AS (
    SELECT item_Id, COUNT(group_Id) AS item_frequency
    FROM recommendations.vw_item_groups
    GROUP BY item_Id
    HAVING item_frequency >= min_item_frequency
  ),

  # Create valid group set
  valid_groups AS (
    SELECT group_Id, COUNT(item_Id) AS group_size
    FROM recommendations.vw_item_groups
    WHERE item_Id IN (SELECT item_Id FROM valid_items)
    GROUP BY group_Id
    HAVING group_size BETWEEN 2 AND max_group_size
  )

  SELECT item_Id, group_Id
  FROM recommendations.vw_item_groups
  WHERE item_Id IN (SELECT item_Id FROM valid_items)
  AND group_Id IN (SELECT group_Id FROM valid_groups);

  # Compute pairwise cooc
  CREATE OR REPLACE TABLE recommendations.item_cooc
  AS
  SELECT item1_Id, item2_Id, SUM(cooc) AS cooc
  FROM
  (
    SELECT
      a.item_Id item1_Id,
      b.item_Id item2_Id,
      1 as cooc
    FROM recommendations.valid_item_groups a
    JOIN recommendations.valid_item_groups b
    ON a.group_Id = b.group_Id
    AND a.item_Id < b.item_Id
  )
  GROUP BY  item1_Id, item2_Id;

  ###################################
  
  # Compute item frequencies
  CREATE OR REPLACE TABLE recommendations.item_frequency
  AS
  SELECT item_Id, COUNT(group_Id) AS frequency
  FROM recommendations.valid_item_groups
  GROUP BY item_Id;

  ###################################
  
  # Compute total frequency |D|
  SET total = (
    SELECT SUM(frequency)  AS total
    FROM recommendations.item_frequency
  );
  
  ###################################
  
  # Add same item frequency as cooc
  CREATE OR REPLACE TABLE recommendations.item_cooc
  AS
  SELECT item1_Id, item2_Id, cooc 
  FROM recommendations.item_cooc
  UNION ALL
  SELECT item_Id as item1_Id, item_Id AS item2_Id, frequency as item_cooc
  FROM recommendations.item_frequency;

  ###################################

  # Create negative samples
  IF negative_sample_size > 0 THEN
    CREATE OR REPLACE TABLE recommendations.item_cooc
    AS

    WITH 
    ordered_items AS (
      SELECT  ROW_NUMBER() OVER (ORDER BY frequency DESC) number, item_Id
      FROM recommendations.item_frequency
    ),

    top_items AS (
      SELECT item_Id
      FROM ordered_items
      WHERE number <= negative_sample_size
    ),

    negative_samples AS (
      SELECT
        a.item as item1_Id,
        b.item as item2_Id,
        1 as cooc
      FROM top_items a
      JOIN top_items b
      ON a.item_Id < b.item_Id
    ),

    merged AS (
      SELECT item1_Id, item2_Id, cooc
      FROM recommendations.item_cooc
      UNION ALL
      SELECT item1_Id, item2_Id, cooc
      FROM negative_samples
    )

    SELECT item1_Id, item2_Id, MAX(cooc) AS cooc
    FROM merged
    GROUP BY item1_Id, item2_Id;
  END IF;
  ###################################
  
  # Compute PMI
  CREATE OR REPLACE TABLE recommendations.item_cooc
  AS
  SELECT
    a.item1_Id,
    a.item2_Id,
    a.cooc,
    LOG(a.cooc, 2) - LOG(b.frequency, 2) - LOG(c.frequency, 2) + LOG(total, 2) AS pmi
  FROM recommendations.item_cooc a
  JOIN recommendations.item_frequency b
  ON a.item1_Id = b.item_Id
  JOIN recommendations.item_frequency c
  ON a.item2_Id = c.item_Id; 
END