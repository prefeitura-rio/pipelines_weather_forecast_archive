# -*- coding: utf-8 -*-
from enum import Enum


class constants(Enum):
    ######################################
    # Automatically managed,
    # please do not change these values
    ######################################
    # Docker image
    DOCKER_TAG = "AUTO_REPLACE_DOCKER_TAG"
    DOCKER_IMAGE_NAME = "AUTO_REPLACE_DOCKER_IMAGE"
    DOCKER_IMAGE = f"{DOCKER_IMAGE_NAME}:{DOCKER_TAG}"
    GCS_FLOWS_BUCKET = "datario-public"

    ######################################
    # Agent labels
    ######################################
    WEATHER_FORECAST_AGENT_LABEL = "weather-forecast"

    ######################################
    # Other constants
    ######################################
    INFISICAL_PATH = "/gypscie_dexl"
    INFISICAL_URL = "URL"
    INFISICAL_USERNAME = "USERNAME"
    INFISICAL_PASSWORD = "PASSWORD"
