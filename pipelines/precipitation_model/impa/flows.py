# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Download sattelite goes 16 data, treat then and predict
"""

from prefect import Parameter  # pylint: disable=E0611, E0401
from prefect.executors import LocalDaskExecutor  # pylint: disable=E0611, E0401
from prefect.run_configs import KubernetesRun  # pylint: disable=E0611, E0401
from prefect.storage import GCS  # pylint: disable=E0611, E0401

# from google.api_core.exceptions import Forbidden
from prefeitura_rio.pipelines_utils.custom import Flow  # pylint: disable=E0611, E0401

# from prefeitura_rio.pipelines_utils.logging import log
# pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.state_handlers import (
    handler_initialize_sentry,
    handler_inject_bd_credentials,
)

from pipelines.constants import constants  # pylint: disable=E0611, E0401
from pipelines.precipitation_model.impa.schedules import (  # pylint: disable=E0611, E0401
    prediction_schedule,
)

# from pipelines.precipitation_model.impa.src.eval.viz.plot_real_time import create_images
from pipelines.precipitation_model.impa.tasks import (  # pylint: disable=E0611, E0401
    build_dataframe_task,
    download_files_from_s3,
    get_predictions,
    get_relevant_dates_informations,
    get_start_datetime,
    process_satellite_task,
)
from pipelines.tasks import (  # pylint: disable=E0611, E0401
    get_storage_destination,
    upload_files_to_storage,
    unzip_files,
)

# from prefeitura_rio.pipelines_utils.tasks import (  # pylint: disable=E0611, E0401
#     create_table_and_upload_to_gcs,
#     get_now_datetime,
#     task_run_dbt_model_task,
# )


# from pipelines.tasks import task_create_partitions  # pylint: disable=E0611, E0401

# from pipelines.precipitation_model.impa.tasks import (  # pylint: disable=E0611, E0401
# access_api,
# calculate_start_and_end_date,
# create_image,
# query_data_from_gcp,
# register_dataset_on_gypscie,
# task_wait_run,


with Flow(
    name="WEATHER FORECAST: Previsão de Chuva - IMPA",
    state_handlers=[
        handler_initialize_sentry,
        handler_inject_bd_credentials,
    ],
    parallelism=10,
    skip_if_running=False,
) as prediction_previsao_chuva_impa:

    #########################
    #  Define parameters    #
    #########################

    # Model parameters
    start_datetime = Parameter(
        "start_datetime",
        default="2024-10-17 10:00:00",
        required=False,
        # description="Datetime in YYYY-MM-dd HH:mm:ss format, UTC timezone",
    )
    num_workers = Parameter(
        "num_workers",
        default=8,
        required=False,
        # description="Number of workers to use for parallel processing",
    )
    cuda = Parameter("cuda", default=False, required=False)  # description="UseCUDA for prediction"
    # memory = Parameter("memory", default="16Gi", required=False)

    # Parameters for saving data on GCP
    materialize_after_dump = Parameter("materialize_after_dump", default=False, required=False)
    dump_mode = Parameter("dump_mode", default=False, required=False)
    dataset_id = mode_redis = Parameter(
        "dataset_id", default="clima_previsao_chuva", required=False
    )
    table_id = Parameter("table_id", default="modelo_satelite_goes_16_impa", required=False)

    download_base_path = "pipelines/precipitation_model/impa/data/raw/satellite"
    #########################
    #  Start flow           #
    #########################

    # Input arguments (These can be passed via Prefect Parameters or CLI)
    n_historical_days = 1
    dt = get_start_datetime(start_datetime=start_datetime)
    relevant_dts, days_of_year, years, relevant_times = get_relevant_dates_informations(
        dt=dt, n_historical_days=n_historical_days
    )

    unzip_files(
        compressed_files=["pipelines/precipitation_model/impa/model_.zip"],
        destination_folder="pipelines/precipitation_model/impa/",
    )
    # Download data from s3
    downloaded_files_rr = download_files_from_s3(
        product="ABI-L2-RRQPEF",
        relevant_dts=relevant_dts,
        days_of_year=days_of_year,
        years=years,
        relevant_times=relevant_times,
        download_base_path=download_base_path,
    )
    downloaded_files_achaf = download_files_from_s3(
        product="ABI-L2-ACHAF",
        relevant_dts=relevant_dts,
        days_of_year=days_of_year,
        years=years,
        relevant_times=relevant_times,
        download_base_path=download_base_path,
    )

    data_processed_rr = process_satellite_task(
        year=years[0],
        day_of_year=days_of_year[0],
        num_workers=num_workers,
        product="ABI-L2-RRQPEF",
        n_historical_days=n_historical_days,
        wait=downloaded_files_rr,
    )
    data_processed_achaf = process_satellite_task(
        year=years[0],
        day_of_year=days_of_year[0],
        num_workers=num_workers,
        product="ABI-L2-ACHAF",
        n_historical_days=n_historical_days,
        wait=[downloaded_files_achaf],
        # wait=[downloaded_files_achaf, data_processed_rr],
    )
    dfr = build_dataframe_task(num_workers, dt, wait=[data_processed_rr, data_processed_achaf])
    output_predict_filepaths = get_predictions(num_workers=num_workers, cuda=cuda, wait=dfr)

    destination_folder_models = get_storage_destination(
        path="cor-clima-imagens/previsao_chuva/impa/modelos"
    )
    upload_files_to_storage(
        project="datario",
        bucket_name="datario-public",
        destination_folder=destination_folder_models,
        source_file_names=output_predict_filepaths,
    )
    # prediction_images_path = create_images()
    # destination_folder_images = get_storage_destination(
    #     path="cor-clima-imagens/previsao_chuva/impa/"
    # )
    # upload_files_to_storage(
    #     project="datario",
    #     bucket_name="datario-public",
    #     destination_folder=destination_folder_images,
    #     source_file_names=prediction_images_path,
    # )

    # image_path = create_image(geolocalized_prediction_datasets)
    # # Save prediction on file
    # prediction_data_path = task_create_partitions(
    #     geolocalized_prediction_datasets,
    #     partition_date_column="data_predicao",  # TODO: change column name
    #     # partition_columns=["ano_particao", "mes_particao", "data_particao"],
    #     savepath="model_prediction",
    #     suffix=now_datetime,
    # )

    # ##############################
    # #  Save predictions on GCP   #
    # ##############################

    # # Upload data to BigQuery
    # create_table = create_table_and_upload_to_gcs(
    #     data_path=prediction_data_path,
    #     dataset_id=dataset_id,
    #     table_id=table_id,
    #     dump_mode=dump_mode,
    #     biglake_table=False,
    # )

    # # Trigger DBT flow run
    # with case(materialize_after_dump, True):
    #     run_dbt = task_run_dbt_model_task(
    #         dataset_id=dataset_id,
    #         table_id=table_id,
    #         # mode=materialization_mode,
    #         # materialize_to_datario=materialize_to_datario,
    #     )
    #     run_dbt.set_upstream(create_table)


##############################
#  Flow run parameters       #
##############################

prediction_previsao_chuva_impa.state_handlers = [handler_inject_bd_credentials]
prediction_previsao_chuva_impa.storage = GCS(constants.GCS_FLOWS_BUCKET.value)
prediction_previsao_chuva_impa.run_config = KubernetesRun(
    image=constants.DOCKER_IMAGE.value,
    labels=[
        constants.WEATHER_FORECAST_AGENT_LABEL.value,
    ],
    cpu_request="500m",
    memory_limit="30Gi",
    memory_request="15Gi",
)
prediction_previsao_chuva_impa.schedule = prediction_schedule
prediction_previsao_chuva_impa.executor = LocalDaskExecutor(num_workers=50)
