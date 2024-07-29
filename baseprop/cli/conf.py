import logging
import os
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(os.getenv("BASEPROP_LOG_DIR", "baseprop_logs"))
LOG_LEVELS = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
NOW = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
