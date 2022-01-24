CREATE OR REPLACE MODEL retail_banking.fraud_prediction
    OPTIONS(model_type='logistic_reg', labels=['is_fraud']) AS
        SELECT * EXCEPT(trans_id)
        FROM retail_banking.training_data
        -- Account for class imbalance. Alternatively, use AUTO_CLASS_WEIGHTS=True in the model options
        WHERE (is_fraud IS TRUE) OR
        (is_fraud IS NOT TRUE 
            AND rand() <=(
                SELECT COUNTIF(is_fraud)/COUNT(*) FROM retail_banking.training_data
        )
);