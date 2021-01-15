CREATE OR REPLACE PROCEDURE @DATASET_NAME.sp_TrainItemMatchingModel(
  IN dimensions INT64
)

BEGIN

  CREATE OR REPLACE MODEL @DATASET_NAME.item_matching_model
  OPTIONS(
    MODEL_TYPE='matrix_factorization', 
    FEEDBACK_TYPE='implicit',
    WALS_ALPHA=1,
    NUM_FACTORS=(dimensions),
    USER_COL='item1_Id', 
    ITEM_COL='item2_Id',
    RATING_COL='score',
    DATA_SPLIT_METHOD='no_split'
  )
  AS
  SELECT 
    item1_Id, 
    item2_Id, 
    cooc *  pmi AS score
  FROM @DATASET_NAME.item_cooc;

END