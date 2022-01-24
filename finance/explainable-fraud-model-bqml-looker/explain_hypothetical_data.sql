SELECT * FROM 
    ML.EXPLAIN_PREDICT(MODEL retail_banking.fraud_prediction, (
            SELECT 
                '001' as trans_id, 
                500.00 as card_transactions_amount, 
                600 as card_transactions_transaction_distance, 
                2 as card_transactions_transaction_hour_of_day
        UNION ALL
            SELECT 
                '002' as trans_id, 
                5.25 as card_transactions_amount, 
                2 as card_transactions_transaction_distance, 
                13 as card_transactions_transaction_hour_of_day
        UNION ALL
            SELECT 
                '003' as trans_id, 
                175.50 as card_transactions_amount, 
                45 as card_transactions_transaction_distance,
                10 as card_transactions_transaction_hour_of_day
    ), STRUCT(0.55 AS threshold)
)