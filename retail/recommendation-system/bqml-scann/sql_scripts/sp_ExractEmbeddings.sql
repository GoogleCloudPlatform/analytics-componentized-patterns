CREATE OR REPLACE PROCEDURE recommendations.sp_ExractEmbeddings() 
BEGIN
CREATE OR REPLACE TABLE  item_recommendations.item_embeddings AS
SELECT 
  feature AS item_Id,
  processed_input AS axis,
  factor_weights,
  intercept
FROM
  ML.WEIGHTS(MODEL `item_recommendations.item_embedding_cooc`)
WHERE feature != 'global__INTERCEPT__';
END