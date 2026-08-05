"""Configuration for the Step 0 agentic disaster monitor."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()

TARGET_MIN = int(os.environ.get("DISASTERRES_AGENT_TARGET_MIN", "100"))
DAILY_SEARCH_CAP = int(os.environ.get("DISASTERRES_TWITTER_DAILY_SEARCH_CAP", "100"))
TWITTER_SEARCH_DELAY_SECONDS = float(os.environ.get("DISASTERRES_TWITTER_SEARCH_DELAY_SECONDS", "5.1"))
NOMINATIM_DELAY_SECONDS = float(os.environ.get("DISASTERRES_NOMINATIM_DELAY_SECONDS", "1.1"))
RAW_DATASET_DIR = os.environ.get("DISASTERRES_RAW_DATASET_DIR", "raw_dataset")
RESULTS_DIR = os.environ.get("DISASTERRES_RESULTS_DIR", "results")
CACHE_PATH = os.environ.get("DISASTERRES_AGENT_CACHE_PATH", os.path.join(".cache", "step0_agent_cache.sqlite3"))
PROVENANCE_PATH = os.path.join(RAW_DATASET_DIR, "step0_provenance.csv")
REACT_TRACE_PATH = os.path.join(RESULTS_DIR, "step0_react_trace.log")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "DisasterRes-Net"
MLFLOW_RUN_NAME = "step0_agent_monitor"


@dataclass(frozen=True)
class TwikitCredentials:
    username: str
    email: str
    password: str


def load_twikit_credentials() -> Optional[TwikitCredentials]:
    username = os.environ.get("TWIKIT_USERNAME")
    email = os.environ.get("TWIKIT_EMAIL")
    password = os.environ.get("TWIKIT_PASSWORD")
    if not username or not email or not password:
        return None
    return TwikitCredentials(username=username, email=email, password=password)

