# Real-time Item-to-item Recommendation BigQuery ML Matrix Factorization and ScaNN

This directory contains a sample code for training and serving embeddings 
for real-time similarity matching. The system utilizes [BigQuery ML Matrix Factorization](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-matrix-factorization)
model to train the embeddings, and the open-source [ScaNN framework](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html) to build and
approximate nearest neighbour index.

1. Compute pointwise mutual information (PMI) between items based on their cooccurrences.
2. Train item embeddings using BigQuery ML Matrix Factorization, using item PMI as implicit feedback.
3. Export and post-process the embeddings from BigQuery ML model to Cloud Storage as CSV files using Cloud Dataflow.
4. Implement an embedding lookup model using Keras and deploy it to AI Platform Prediction.
5. Serve the embedding as an approximate nearest neighbor index using ScaNN

![Workflow](diagram.png)