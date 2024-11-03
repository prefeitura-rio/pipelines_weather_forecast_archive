# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Download meteorological data, treat then, integrate and predict
"""

from prefect import Parameter, case  # pylint: disable=E0611, E0401
from prefect.run_configs import KubernetesRun  # pylint: disable=E0611, E0401
from prefect.storage import GCS  # pylint: disable=E0611, E0401

# from google.api_core.exceptions import Forbidden
from prefeitura_rio.pipelines_utils.custom import Flow  # pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.state_handlers import (
    handler_initialize_sentry,
    handler_inject_bd_credentials,
)
from prefeitura_rio.pipelines_utils.tasks import (  # pylint: disable=E0611, E0401
    get_now_datetime,
)

# from prefeitura_rio.pipelines_utils.tasks import (  # pylint: disable=E0611, E0401;
#     create_table_and_upload_to_gcs,
#     task_run_dbt_model_task,
# )
from pipelines.constants import constants  # pylint: disable=E0611, E0401
from pipelines.precipitation_model.rionowcast.schedules import (  # pylint: disable=E0611, E0401
    prediction_schedule,
)
from pipelines.precipitation_model.rionowcast.tasks import (  # pylint: disable=E0611, E0401
    access_api,
    download_datasets_from_gypscie,
    execute_dataset_processor,
    execute_prediction_on_gypscie,
    get_dataflow_params,
    get_dataset_info,
    get_dataset_name_on_gypscie,
    get_dataset_processor_info,
    query_data_from_gcp,
    read_numpy_files,
    register_dataset_on_gypscie,
    task_wait_run,
    unzip_files,
)

# get_billing_project_id, add_columns_on_dfr, create_image, desnormalize_data,
# geolocalize_data, path_to_dfr,

# from pipelines.tasks import (  # pylint: disable=E0611, E0401
#     get_storage_destination,
#     task_create_partitions,
#     upload_files_to_storage,
# )

with Flow(
    name="WEATHER FORECAST: Pré-processamento dos dados - Rionowcast",
    state_handlers=[
        handler_initialize_sentry,
        handler_inject_bd_credentials,
    ],
    parallelism=10,
    skip_if_running=False,
) as preprocessing_previsao_chuva_rionowcast:

    # Parameters to run a query on Bigquery
    bd_project_mode = Parameter("bd_project_mode", default="prod", required=False)
    billing_project_id = Parameter("billing_project_id", default="rj-cor", required=False)
    billing_project_id = "rj-cor"

    # Query parameters
    data_type = Parameter("data_type", default=None, required=False)
    start_date = Parameter("start_date", default=None, required=False)
    end_date = Parameter("end_date", default=None, required=False)

    # Gypscie parameters
    environment_id = Parameter("environment_id", default=1, required=False)
    domain_id = Parameter("domain_id", default=1, required=False)
    project_id = Parameter("project_id", default=1, required=False)
    project_name = Parameter("project_name", default="rionowcast_precipitation", required=False)

    # Gypscie processor parameters
    processor_name = Parameter("processor_name", default="etl_alertario22", required=True)
    dataset_processor_id = Parameter("dataset_processor_id", default=43, required=False)  # mudar

    # Parameters for saving data on GCP
    materialize_after_dump = Parameter("materialize_after_dump", default=False, required=False)
    dump_mode = Parameter("dump_mode", default=False, required=False)
    dataset_id = Parameter("dataset_id", default="clima_previsao_chuva", required=False)
    table_id = Parameter(
        "table_id", default="modelo_pluviometro_alertario_radar_mendanha_rionowcast", required=False
    )

    # Dataset parameters
    station_type = Parameter("station_type", default="pluviometro", required=False)
    source = Parameter("source", default="alertario", required=False)

    # Dataset path, if it was saved on ETL flow or it will be None
    dataset_path = Parameter("dataset_path", default=None, required=False)  # dataset_path
    model_version = Parameter("model_version", default=1, required=False)

    ####################################
    #  Start preprocessing flow        #
    ####################################

    api = access_api()

    dataset_info = get_dataset_info(station_type, source)

    # Get data from GCP if you don't have a path
    with case(dataset_path, None):
        with case(station_type, not "radar"):
            # billing_project_id = get_billing_project_id(bd_project_mode, billing_project_id)
            dataset_path = query_data_from_gcp(
                dataset_info["dataset_id"],
                dataset_info["table_id"],
                billing_project_id="rj-cor",
                start_date=start_date,
                end_date=end_date,
                save_format="parquet",
            )

        with case(station_type, "radar"):
            dataset_path = None
            # TODO: download data from storage

    # Get processor information on gypscie
    with case(dataset_processor_id, None):
        dataset_processor_response, dataset_processor_id = get_dataset_processor_info(
            api, processor_name
        )

    dataset_response = register_dataset_on_gypscie(api, filepath=dataset_path, domain_id=domain_id)
    # TODO: verifcar no codigo do augustp se são esses os parametros corretos
    processor_parameters = {
        "dataset1": str(dataset_path).rsplit("/", maxsplit=1)[-1],
        "station_type": station_type,
    }

    dataset_processor_task_id = execute_dataset_processor(
        api,
        processor_id=dataset_processor_id,
        dataset_id=[dataset_response["id"]],
        environment_id=environment_id,
        project_id=project_id,
        parameters=processor_parameters,
    )
    wait_run = task_wait_run(api, dataset_processor_task_id, flow_type="processor")
    dataset_name = get_dataset_name_on_gypscie(api, wait_run["dataset_id"])
    dataset_path = download_datasets_from_gypscie(api, dataset_names=[dataset_name], wait=wait_run)
    # dfr_ = path_to_dfr(path=dataset_path)
    # # output_datasets_id = get_output_dataset_ids_on_gypscie(api, dataset_processor_task_id)
    # dfr = add_columns_on_dfr(dfr_, model_version, update_time=True)

    # # Save pre-treated data on local file with partitions
    # now_datetime = get_now_datetime()
    # prediction_data_path = task_create_partitions(
    #     dfr,
    #     partition_date_column=dataset_info["partition_date_column"],
    #     savepath="model_prediction",
    #     suffix=now_datetime,
    # )
    # ################################
    # #  Save preprocessing on GCP   #
    # ################################

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
preprocessing_previsao_chuva_rionowcast.state_handlers = [handler_inject_bd_credentials]
preprocessing_previsao_chuva_rionowcast.storage = GCS(constants.GCS_FLOWS_BUCKET.value)
preprocessing_previsao_chuva_rionowcast.run_config = KubernetesRun(
    image=constants.DOCKER_IMAGE.value,
    labels=[constants.WEATHER_FORECAST_AGENT_LABEL.value],
)
# preprocessing_previsao_chuva_rionowcast.schedule = update_schedule

# https://github.com/prefeitura-rio/pipelines_rj_escritorio/blob/2433238db27adb1213059832f238495b9ecb5043/pipelines/deteccao_alagamento_cameras/flooding_detection/flows.py#L112
# https://linen.prefect.io/t/13543083/how-do-i-run-the-same-subflow-concurrently-for-items-in-a-li


with Flow(
    name="WEATHER FORECAST: Previsão de Chuva - Rionowcast",
    state_handlers=[
        handler_initialize_sentry,
        handler_inject_bd_credentials,
    ],
    parallelism=10,
    skip_if_running=False,
) as prediction_previsao_chuva_rionowcast:

    #########################
    #  Define parameters    #
    #########################

    # Model parameters
    hours_from_past = Parameter("hours_from_past", required=True, default=6)
    start_date = Parameter("start_date", default=None, required=False)
    end_date = Parameter("end_date", default=None, required=False)

    # Gypscie parameters
    environment_id = Parameter("environment_id", default=1, required=False)
    domain_id = Parameter("domain_id", default=1, required=False)
    project_id = Parameter("project_id", default=1, required=False)

    # Gypscie functions and workflow parameters
    workflow_id = Parameter("workflow_id", default=43, required=False)  # mudar.
    load_data_function_id = Parameter("load_data_function_id", default=59, required=False)  # mudar
    pre_processing_function_id = Parameter(
        "pre_processing_function_id", default=60, required=False
    )  # mudar.
    model_function_id = Parameter(
        "model_function_id",
        default=61,
        required=False,
        # description="Id of the function of the model",
    )  # mudar.

    # Gypscie dataset parameters
    model_data_id = Parameter(
        "model_data_id",
        default=191,
        required=False,
        # description="Id of the model saved as a dataset",
    )  # mudar.
    output_function_id = Parameter("output_function_id", default=62, required=False)  # mudar.
    # radar_data_id = Parameter("radar_data_id", default=25, required=False)
    # rain_gauge_data_id = Parameter("rain_gauge_data_id", default=22, required=False)
    grid_data_id = Parameter(
        "grid_data_id", default=177, required=False  # , description="Grid ID saved as a dataset"
    )  # mudar.

    # Parameters for saving data on GCP
    materialize_after_dump = Parameter("materialize_after_dump", default=False, required=False)
    dump_mode = Parameter("dump_mode", default=False, required=False)
    dataset_id = mode_redis = Parameter("dataset_id", default="clima_rionowcast", required=False)
    table_id = Parameter("table_id", default="predicao_precipitacao", required=False)()

    model_version = Parameter("model_version", default=1, required=False)

    # Pre-treated Data Sources on GCP
    pluviometer_dataset_info = {  # fonte: Saída do Dataflow ETL de pluviômetros no GCP
        "dataset_id": "clima_rionowcast",
        "table_id": "preprocessamento_pluviometro_alertario",
        "filename": "gauge_station_bq",
    }

    radar_dataset_info = {  # fonte: Saída do Dataflow ETL de radar no GCP
        "dataset_id": "clima_rionowcast",
        "table_id": "preprocessamento_radar_mendanha",
        "filename": "radar_bq",
    }

    #########################
    #  Start flow           #
    #########################

    api = access_api()

    # with case(start_date, None):
    #     start_date, end_date = calculate_start_and_end_date(hours_from_past)

    # # Get data from pre-treated sources that were saved on gcp
    # pluviometer_alertario_path = query_data_from_gcp(
    #     pluviometer_dataset_info["dataset_id"],
    #     pluviometer_dataset_info["table_id"],
    #     billing_project_id="rj-cor",
    #     start_date=start_date,
    #     end_date=end_date,
    #     save_format="parquet",
    # )
    # radar_mendanha_path = query_data_from_gcp(
    #     radar_dataset_info["dataset_id"],
    #     radar_dataset_info["table_id"],
    #     billing_project_id="rj-cor",
    #     start_date=start_date,
    #     end_date=end_date,
    #     save_format="parquet",
    # )

    # # Register these datasets on gypscie
    # pluviometer_alertario_registered = register_dataset_on_gypscie(
    # api, pluviometer_alertario_path)  # noqa E501, C0301
    # radar_mendanha_registered = register_dataset_on_gypscie(api, radar_mendanha_path)

    pluviometer_alertario_registered = {"id": 231}
    radar_mendanha_registered = {"id": 230}

    model_params = get_dataflow_params(
        workflow_id=workflow_id,
        environment_id=environment_id,
        project_id=project_id,
        load_data_funtion_id=load_data_function_id,
        pre_processing_function_id=pre_processing_function_id,
        model_function_id=model_function_id,
        radar_data_id=pluviometer_alertario_registered["id"],
        rain_gauge_data_id=radar_mendanha_registered["id"],
        grid_data_id=grid_data_id,
        model_data_id=model_data_id,
        output_function_id=output_function_id,
    )

    # Send dataset ids to gypscie to get predictions
    output_dataset_ids = execute_prediction_on_gypscie(
        api,
        model_params,
    )
    # prediction_dataset_ids = get_output_dataset_ids_on_gypscie(api, task_id)
    # wait_run = task_wait_run(api, task_id, flow_type="processor")  # new
    dataset_names = get_dataset_name_on_gypscie(api, output_dataset_ids)  # new
    ziped_dataset_paths = download_datasets_from_gypscie(api, dataset_names=dataset_names)
    dataset_paths = unzip_files(ziped_dataset_paths)
    prediction_datasets = read_numpy_files(dataset_paths)

    # prediction_datasets = download_datasets_from_gypscie(api, prediction_dataset_ids)
    # desnormalized_prediction_datasets = desnormalize_data(prediction_datasets)
    now_datetime = get_now_datetime()
    # geolocalized_prediction_datasets = geolocalize_data(
    #     desnormalized_prediction_datasets, now_datetime
    # )
    # images_path_wb = create_image(geolocalized_prediction_datasets)
    # dfr = add_columns_on_dfr(geolocalized_prediction_datasets, model_version)

    # ##############################
    # #  Save image on GCP         #
    # ##############################
    # destination_folder_wb = get_storage_destination(
    #     path="cor-clima-imagens/predicao_precipitacao/rionowcast_1/without_background"
    # )
    # upload_files_to_storage(
    #     project="datario",
    #     bucket_name="datario-public",
    #     destination_folder=destination_folder_wb,
    #     source_file_names=images_path_wb,
    # )

    # ##############################
    # #  Save predictions on GCP   #
    # ##############################

    # # Save prediction on file
    # prediction_data_path = task_create_partitions(
    #     dfr,
    #     partition_date_column="data_predicao",  # TODO: change column name
    #     # partition_columns=["ano_particao", "mes_particao", "data_particao"],
    #     savepath="model_prediction",
    #     suffix=now_datetime,
    # )

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

prediction_previsao_chuva_rionowcast.state_handlers = [handler_inject_bd_credentials]
prediction_previsao_chuva_rionowcast.storage = GCS(constants.GCS_FLOWS_BUCKET.value)
prediction_previsao_chuva_rionowcast.run_config = KubernetesRun(
    image=constants.DOCKER_IMAGE.value,
    labels=[constants.WEATHER_FORECAST_AGENT_LABEL.value],
)
prediction_previsao_chuva_rionowcast.schedule = prediction_schedule
