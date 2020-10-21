# Real-time Item-to-item Recommendation BigQueryML Matrix Factorization and ScaNN

1. Compute pointwise mutual information (PMI) between items based on their cooccurrences.
2. Train item embeddings using BigQuery ML Matrix Factorization, using item PMI as implicit feedback.
3. Export and post-process the embeddings from BigQuery ML model to Cloud Storage as CSV files using Cloud Dataflow.
4. Implement an embedding lookup model using Keras and deploys it to AI Platfor Prediction.
5. Serve the embedding as an approximate nearest neighbor index using ScaNN