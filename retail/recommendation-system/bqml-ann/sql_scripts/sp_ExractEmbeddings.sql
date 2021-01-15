CREATE OR REPLACE PROCEDURE @DATASET_NAME.sp_ExractEmbeddings() 
BEGIN
  CREATE OR REPLACE TABLE  @DATASET_NAME.item_embeddings AS
    WITH 
    step1 AS
    (
        SELECT 
            feature AS item_Id,
            factor_weights,
            intercept AS bias,
        FROM
            ML.WEIGHTS(MODEL `@DATASET_NAME.item_matching_model`)
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
    GROUP BY id
END