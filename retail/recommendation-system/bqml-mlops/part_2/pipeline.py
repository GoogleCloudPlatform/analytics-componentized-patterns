from typing import NamedTuple
import json
import os
import fire

def run_bigquery_ddl(project_id: str, query_string: str, location: str) -> NamedTuple(
    'DDLOutput', [('created_table', str), ('query', str)]):
    """
    Runs BigQuery query and returns a table/model name
    """
    print(query_string)
        
    from google.cloud import bigquery
    from google.api_core.future import polling
    from google.cloud import bigquery
    from google.cloud.bigquery import retry as bq_retry
    
    bqclient = bigquery.Client(project=project_id, location=location)
    job = bqclient.query(query_string, retry=bq_retry.DEFAULT_RETRY)
    job._retry = polling.DEFAULT_RETRY
    
    while job.running():
        from time import sleep
        sleep(0.1)
        print('Running ...')
        
    tblname = job.ddl_target_table
    tblname = '{}.{}'.format(tblname.dataset_id, tblname.table_id)
    print('{} created in {}'.format(tblname, job.ended - job.started))
    
    from collections import namedtuple
    result_tuple = namedtuple('DDLOutput', ['created_table', 'query'])
    return result_tuple(tblname, query_string)

def train_matrix_factorization_model(ddlop, project_id: str, dataset: str):
    query = """
        CREATE OR REPLACE MODEL `{project_id}.{dataset}.my_implicit_mf_model_quantiles_demo_binary_prod`
        OPTIONS
          (model_type='matrix_factorization',
           feedback_type='implicit',
           user_col='user_id',
           item_col='hotel_cluster',
           rating_col='rating',
           l2_reg=30,
           num_factors=15) AS

        SELECT
          user_id,
          hotel_cluster,
          if(sum(is_booking) > 0, 1, sum(is_booking)) AS rating
        FROM `{project_id}.{dataset}.hotel_train`
        group by 1,2
    """.format(project_id = project_id, dataset = dataset)
    return ddlop(project_id, query, 'US')

def evaluate_matrix_factorization_model(project_id:str, mf_model:str, location:str='US')-> NamedTuple('MFMetrics', [('msqe', float)]):
    
    query = """
        SELECT * FROM ML.EVALUATE(MODEL `{project_id}.{mf_model}`)
    """.format(project_id = project_id, mf_model = mf_model)

    print(query)

    from google.cloud import bigquery
    import json

    bqclient = bigquery.Client(project=project_id, location=location)
    job = bqclient.query(query)
    metrics_df = job.result().to_dataframe()
    from collections import namedtuple
    result_tuple = namedtuple('MFMetrics', ['msqe'])
    return result_tuple(metrics_df.loc[0].to_dict()['mean_squared_error'])

def create_user_features(ddlop, project_id:str, dataset:str, mf_model:str):
    #Feature engineering for useres
    query = """
        CREATE OR REPLACE TABLE  `{project_id}.{dataset}.user_features_prod` AS
        WITH u as 
        (
            select
            user_id,
            count(*) as total_visits,
            count(distinct user_location_city) as distinct_cities,
            sum(distinct site_name) as distinct_sites,
            sum(is_mobile) as total_mobile,
            sum(is_booking) as total_bookings,
            FROM `{project_id}.{dataset}.hotel_train`
            GROUP BY 1
        )
        SELECT
            u.*,
            (SELECT ARRAY_AGG(weight) FROM UNNEST(factor_weights)) AS user_factors
        FROM
            u JOIN ML.WEIGHTS( MODEL `{mf_model}`) w
            ON processed_input = 'user_id' AND feature = CAST(u.user_id AS STRING)
    """.format(project_id = project_id, dataset = dataset, mf_model=mf_model)
    return ddlop(project_id, query, 'US')

def create_hotel_features(ddlop, project_id:str, dataset:str, mf_model:str):
    #Feature eingineering for hotels
    query = """
        CREATE OR REPLACE TABLE  `{project_id}.{dataset}.hotel_features_prod` AS
        WITH h as 
        (
            select
            hotel_cluster,
            count(*) as total_cluster_searches,
            count(distinct hotel_country) as distinct_hotel_countries,
            sum(distinct hotel_market) as distinct_hotel_markets,
            sum(is_mobile) as total_mobile_searches,
            sum(is_booking) as total_cluster_bookings,
            FROM `{project_id}.{dataset}.hotel_train`
            group by 1
        )
        SELECT
            h.*,
            (SELECT ARRAY_AGG(weight) FROM UNNEST(factor_weights)) AS hotel_factors
        FROM
            h JOIN ML.WEIGHTS( MODEL `{mf_model}`) w
            ON processed_input = 'hotel_cluster' AND feature = CAST(h.hotel_cluster AS STRING)
    """.format(project_id = project_id, dataset = dataset, mf_model=mf_model)
    return ddlop(project_id, query, 'US')

def combine_features(ddlop, project_id:str, dataset:str, mf_model:str, hotel_features:str, user_features:str):
    #Combine user and hotel embedding features with the rating associated with each combination
    query = """
        CREATE OR REPLACE TABLE `{project_id}.{dataset}.total_features_prod` AS
        with ratings as(
          SELECT
            user_id,
            hotel_cluster,
            if(sum(is_booking) > 0, 1, sum(is_booking)) AS rating
          FROM `{project_id}.{dataset}.hotel_train`
          group by 1,2
        )
        select
          h.* EXCEPT(hotel_cluster),
          u.* EXCEPT(user_id),
          IFNULL(rating,0) as rating
        from `{hotel_features}` h, `{user_features}` u
        LEFT OUTER JOIN ratings r
        ON r.user_id = u.user_id AND r.hotel_cluster = h.hotel_cluster
    """.format(project_id = project_id, dataset = dataset, mf_model=mf_model, hotel_features=hotel_features, user_features=user_features)
    return ddlop(project_id, query, 'US')

def train_xgboost_model(ddlop, project_id:str, dataset:str, total_features:str):
    #Combine user and hotel embedding features with the rating associated with each combination
    query = """
        CREATE OR REPLACE MODEL `{project_id}.{dataset}.recommender_hybrid_xgboost_prod` 
        OPTIONS(model_type='boosted_tree_classifier', input_label_cols=['rating'], AUTO_CLASS_WEIGHTS=True)
        AS

        SELECT
          * EXCEPT(user_factors, hotel_factors),
            {dataset}.arr_to_input_15_users(user_factors).*,
            {dataset}.arr_to_input_15_hotels(hotel_factors).*
        FROM
          `{total_features}`
    """.format(project_id = project_id, dataset = dataset, total_features=total_features)
    return ddlop(project_id, query, 'US')

def evaluate_class(project_id:str, dataset:str, class_model:str, total_features:str, location:str='US')-> NamedTuple('ClassMetrics', [('roc_auc', float)]):     
    query = """
        SELECT
          *
        FROM ML.EVALUATE(MODEL `{class_model}`, (
          SELECT
          * EXCEPT(user_factors, hotel_factors),
            {dataset}.arr_to_input_15_users(user_factors).*,
            {dataset}.arr_to_input_15_hotels(hotel_factors).*
        FROM
          `{total_features}`
        ))
    """.format(dataset = dataset, class_model = class_model, total_features = total_features)

    print(query)

    from google.cloud import bigquery

    bqclient = bigquery.Client(project=project_id, location=location)
    job = bqclient.query(query)
    metrics_df = job.result().to_dataframe()
    from collections import namedtuple
    result_tuple = namedtuple('ClassMetrics', ['roc_auc'])
    return result_tuple(metrics_df.loc[0].to_dict()['roc_auc'])

def export_bqml_model(project_id:str, model:str, destination:str) -> NamedTuple('ModelExport', [('destination', str)]):
    import subprocess
    #command='bq extract -destination_format=ML_XGBOOST_BOOSTER -m {}:{} {}'.format(project_id, model, destination)
    model_name = '{}:{}'.format(project_id, model)
    print (model_name)
    subprocess.run(['bq', 'extract', '-destination_format=ML_XGBOOST_BOOSTER', '-m', model_name, destination], check=True)
    from collections import namedtuple
    result_tuple = namedtuple('ModelExport', ['destination'])
    return result_tuple(destination)

import kfp.dsl as dsl
import kfp.components as comp
import time

@dsl.pipeline(
    name='Training pipeline for hotel recommendation prediction',
    description='Training pipeline for hotel recommendation prediction'
)
def training_pipeline(project_id:str, dataset_name:str, model_storage:str, base_image_path:str):

    import json
        
    #############################                                 
    #Defining pipeline execution graph
    dataset = dataset_name
    base_image = 'YOUR BASE IMAGE CONTAINER URL FOR COMPONENTS BELOW'
    
    #Minimum threshold for model metric to determine if model will be deployed to inference: 0.5 is a basically a coin toss with 50/50 chance
    mf_msqe_threshold = 0.5
    class_auc_threshold = 0.8
    
    #Defining function containers
    ddlop = comp.func_to_container_op(run_bigquery_ddl, packages_to_install=['google-cloud-bigquery'])
    evaluate_mf_op = comp.func_to_container_op(evaluate_matrix_factorization_model, base_image=base_image, packages_to_install=['google-cloud-bigquery','pandas'])
    evaluate_class_op = comp.func_to_container_op(evaluate_class, base_image=base_image, packages_to_install=['google-cloud-bigquery','pandas'])
    export_bqml_model_op = comp.func_to_container_op(export_bqml_model, base_image=base_image, packages_to_install=['google-cloud-bigquery'])
    
    #Train matrix factorization model
    mf_model_output = train_matrix_factorization_model(ddlop, project_id, dataset).set_display_name('train matrix factorization model')
    mf_model_output.execution_options.caching_strategy.max_cache_staleness = 'P0D'
    mf_model = mf_model_output.outputs['created_table']
    
    #Evaluate matrix factorization model
    mf_eval_output = evaluate_mf_op(project_id, mf_model).set_display_name('evaluate matrix factorization model')
    mf_eval_output.execution_options.caching_strategy.max_cache_staleness = 'P0D'
    
    #mean squared quantization error 
    with dsl.Condition(mf_eval_output.outputs['msqe'] < mf_msqe_threshold):
    
        #Create features for Classification model
        user_features_output = create_user_features(ddlop, project_id, dataset, mf_model).set_display_name('create user factors features')
        user_features = user_features_output.outputs['created_table']
        user_features_output.execution_options.caching_strategy.max_cache_staleness = 'P0D'
        
        hotel_features_output = create_hotel_features(ddlop, project_id, dataset, mf_model).set_display_name('create hotel factors features')
        hotel_features = hotel_features_output.outputs['created_table']
        hotel_features_output.execution_options.caching_strategy.max_cache_staleness = 'P0D'

        total_features_output = combine_features(ddlop, project_id, dataset, mf_model, hotel_features, user_features).set_display_name('combine all features')
        total_features = total_features_output.outputs['created_table']
        total_features_output.execution_options.caching_strategy.max_cache_staleness = 'P0D'

        #Train XGBoost model
        class_model_output = train_xgboost_model(ddlop, project_id, dataset, total_features).set_display_name('train XGBoost model')
        class_model = class_model_output.outputs['created_table']
        class_model_output.execution_options.caching_strategy.max_cache_staleness = 'P0D'

        class_model = 'hotel_recommendations.recommender_hybrid_xgboost_prod'
    
        #Evaluate XGBoost model
        class_eval_output = evaluate_class_op(project_id, dataset, class_model, total_features).set_display_name('evaluate XGBoost model')
        class_eval_output.execution_options.caching_strategy.max_cache_staleness = 'P0D'
        
        with dsl.Condition(class_eval_output.outputs['roc_auc'] > class_auc_threshold):
            #Export model
            export_destination_output = export_bqml_model_op(project_id, class_model, model_storage).set_display_name('export XGBoost model')
            export_destination_output.execution_options.caching_strategy.max_cache_staleness = 'P0D'
            export_destination = export_destination_output.outputs['destination']

def main(**args):
    #Specify pipeline argument values
    arguments = {
        'project_id': args['project_id'],
        'dataset_name': args['dataset_name'],
        'model_storage': args['model_storage']
    }
    
    pipeline_func = training_pipeline
    pipeline_filename = pipeline_func.__name__ + '.zip'
    import kfp.compiler as compiler
    import kfp
    compiler.Compiler().compile(pipeline_func, pipeline_filename)


    #Get or create an experiment and submit a pipeline run
    client = kfp.Client(args['kfp_host'])
    experiment = client.create_experiment('hotel_recommender_experiment')

    #Submit a pipeline run
    run_name = pipeline_func.__name__ + ' run'
    run_result = client.run_pipeline(experiment.id, run_name, pipeline_filename, arguments)
    
if __name__ == '__main__':
  fire.Fire(main)