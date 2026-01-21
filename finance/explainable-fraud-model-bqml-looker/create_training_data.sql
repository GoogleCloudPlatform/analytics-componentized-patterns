
CREATE OR REPLACE TABLE retail_banking.training_data as (
    SELECT
        card_transactions.trans_id AS trans_id,
        card_transactions.is_fraud AS is_fraud,
        --amount for transaction: higher amounts are more likely to be fraud
        cast(card_transactions.amount as FLOAT64) AS card_transactions_amount,

        --distance from the customers home: further distances are more likely to be fraud
        ST_DISTANCE((ST_GEOGPOINT((cast(card_transactions.merchant_lon as FLOAT64)),
                (cast(card_transactions.merchant_lat as FLOAT64)))), 
            (ST_GeogPoint((cast(SPLIT(client.address,'|')[
                OFFSET
                    (4)] as float64)),
                (cast(SPLIT(client.address,'|')[
                    OFFSET
                    (3)] as float64))))) AS card_transactions_transaction_distance,

        --hour that transaction occured: fraud occurs in middle of night (usually between midnight and 4 am)
        EXTRACT(HOUR FROM TIMESTAMP(CONCAT(card_transactions.trans_date,' ',card_transactions.trans_time)) ) AS card_transactions_transaction_hour_of_day
    
    FROM `looker-private-demo.retail_banking.card_transactions` AS card_transactions
    LEFT JOIN `looker-private-demo.retail_banking.card` AS card 
        ON card.card_number = card_transactions.cc_number
    LEFT JOIN `looker-private-demo.retail_banking.disp` AS disp 
        ON card.disp_id = disp.disp_id
    LEFT JOIN `looker-private-demo.retail_banking.client`AS client 
        ON disp.client_id = client.client_id       
);