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
            interval=timedelta(minutes=20),
            start_date=datetime(2023, 1, 1, 0, 1, 0),
            labels=[
                constants.WEATHER_FORECAST_AGENT_LABEL.value,
            ],
            parameter_defaults={
                # "trigger_rain_dashboard_update": True,
                "materialize_after_dump": False,
                "mode": "prod",
                "materialize_to_datario": False,
                "dump_to_gcs": False,
                "dump_mode": "append",
                # "dataset_id": "clima_pluviometro",
                # "table_id": "taxa_precipitacao_cemaden",
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
