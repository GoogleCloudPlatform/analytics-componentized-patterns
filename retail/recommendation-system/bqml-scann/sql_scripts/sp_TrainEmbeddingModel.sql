CREATE OR REPLACE PROCEDURE recommendations.sp_TrainEmbeddingModel(
  IN dimensions INT64
)

BEGIN

  CREATE OR REPLACE MODEL recommendations.item_embedding_model
  OPTIONS(
    MODEL_TYPE='matrix_factorization', 
    FEEDBACK_TYPE='implicit',
    WALS_ALPHA=1,
    NUM_FACTORS=(dimensions),
    USER_COL='item1_Id', 
    ITEM_COL='item2_Id',
    RATING_COL='target',
    DATA_SPLIT_METHOD='no_split'
  )
  AS
  SELECT 
    item1_Id, 
    item2_Id, 
    cooc AS target
  FROM recommendations.item_cooc;

END