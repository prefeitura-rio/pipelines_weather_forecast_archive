# -*- coding: utf-8 -*-
"""
Schedules for the database dump pipeline
"""

from datetime import datetime, timedelta

from prefect.schedules import Schedule
from prefect.schedules.clocks import IntervalClock

from pipelines.constants import constants

update_schedule = Schedule(
    clocks=[
        IntervalClock(
            interval=timedelta(minutes=60),
            start_date=datetime(2023, 1, 1, 0, 1, 0),
            labels=[
                constants.WEATHER_FORECAST_AGENT_LABEL.value,
            ],
            parameter_defaults={
                "dataset_id": "clima_rionowcast",
                "table_id": "predicao_precipitacao",
                "materialize_after_dump": False,
                "mode": "prod",
                "materialize_to_datario": False,
                "dump_to_gcs": False,
                "dump_mode": "append",
                "hours_from_past": 6,
                "end_historical_datetime_date": None,
                "environment_id": 1,
                "domain_id": 1,
                "project_id": 1,
                "workflow_id": 36,
                "load_data_function_id": 42,
                "pre_processing_function_id": 43,
                "model_function_id": 2,
                "model_data_id": 24,
                "output_function_id": 62,
                "grid_data_id": 25,
                # "radar_data_id": None,
                # "rain_gauge_data_id": None,
                "model_version": 1,
            },
        ),
    ]
)

prediction_schedule = Schedule(
    clocks=[
        IntervalClock(
            interval=timedelta(minutes=60),
            start_date=datetime(2023, 1, 1, 0, 1, 0),
            labels=[
                constants.WEATHER_FORECAST_AGENT_LABEL.value,
            ],
            parameter_defaults={
                # "trigger_rain_dashboard_update": True,
                "materialize_after_dump": False,
                "mode": "prod",
                "dump_to_gcs": False,
                "dump_mode": "append",
                # "dataset_id": "clima_pluviometro",
                # "table_id": "taxa_precipitacao_cemaden",
            },
        ),
    ]
)
