import ast
import json
import logging
import os
import sys

import pandas as pd
from loguru import logger as loguru_logger
from pathlib import Path

# Configure the root logger
logging.getLogger("googleapiclient.discovery").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# DEFAULT_LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "logs.jsonl")

DEFAULT_LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "logs.jsonl"

# Make if logs directory doesn't exist
DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class LogManager:
    _instance = None

    def __new__(cls, log_file=DEFAULT_LOG_FILE, **kwargs):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        log_file: str = DEFAULT_LOG_FILE,
        stdout_level: str = "INFO",
        file_level: str = "DEBUG",
        rotation: str = "1 day",
        retention: str = "5 days",
    ):
        if self._initialized:
            return
        self.log_file = os.path.abspath(log_file)
        self.log_name: str = os.path.basename(self.log_file)
        self.stdout_level = stdout_level
        self.file_level = file_level
        self.rotation = rotation
        self.retention = retention
        self._setup_logger()
        self._initialized = True

    def _setup_logger(self):
        def serialize(record) -> str:
            subset = {
                "created_date": record["time"].timestamp(),
                "level": record["level"].name,
                "module": record["module"],
                "filename": record["file"].name,
                "function": record["function"],
                "line": record["line"],
                "message": record["message"],
            }
            return json.dumps(subset)

        def patching(record) -> dict:
            record["extra"]["serialized"] = serialize(record)
            return record

        loguru_logger.remove()
        self.logger = loguru_logger.patch(patching)

        self.logger.add(
            sys.stdout,
            format="<level>{time:YYYY-DD-MM hh:mm:ss A} {level} {message}</level>",
            level=self.stdout_level,
            enqueue=True,
        )

        self.logger.add(
            self.log_file,
            level=self.file_level,
            rotation=self.rotation,
            retention=self.retention,
            serialize=True,
            format="{extra[serialized]}",
        )

    def get_logs(
        self, filter: str = None, show_everything: bool = False
    ) -> pd.DataFrame:

        logs = []
        with open(self.log_file, "r") as f:
            for line in f:
                line = json.loads(line)
                record = line.get("record")
                level = record.get("level")
                name = level.get("name")

                if filter == None:
                    logs.append(record)
                else:
                    if name == filter:
                        logs.append(record)

        if not logs:
            self.logger.info("No logs found")
            return pd.DataFrame()

        df = pd.json_normalize(logs, sep="_")

        if show_everything:
            return df

        # Filter for useful columns only
        slim = df["extra_serialized"].apply(ast.literal_eval).apply(pd.Series)

        # Convert created date timestamp
        slim["created_date"] = pd.to_datetime(
            slim["created_date"], unit="s", utc=True
        ).dt.tz_convert("America/New_York")

        # Sort by date
        slim.sort_values("created_date", ascending=False, inplace=True)

        # Format date
        slim["created_date"] = slim["created_date"].dt.strftime("%Y-%m-%d %I:%M:%S %p")

        return slim.reset_index(drop=True)

    def __getattr__(self, name):
        return getattr(self.logger, name)

    def erase_logs(self):
        with open(self.log_file, "w") as f:
            f.write("")
        self.logger.info(f"Erased logs from {self.log_name}")

    def intercept_package_logs(self, package_name: str, level: str = "INFO"):
        """
        Intercept logs from a specific package and route them through the LogManager.

        Args:
            package_name (str): The name of the package to intercept logs from.
            level (str): The minimum log level to intercept (default: "INFO").
        """

        class InterceptHandler(logging.Handler):
            def emit(self, record):
                # Get corresponding Loguru level if it exists
                try:
                    level = logger.level(record.levelname).name
                except ValueError:
                    level = record.levelno

                # Find caller from where originated the logged message
                frame, depth = logging.currentframe(), 2
                while frame.f_code.co_filename == logging.__file__:
                    frame = frame.f_back
                    depth += 1

                logger.opt(depth=depth, exception=record.exc_info).log(
                    level, record.getMessage()
                )

        logging.getLogger(package_name).handlers = [InterceptHandler()]
        logging.getLogger(package_name).setLevel(level)
        self.logger.info(f"Intercepting logs from package: {package_name}")


# Create a single instance of LogManager
logger = LogManager(stdout_level="INFO", rotation=None, retention=None)


def get_logger():
    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.info("This is an info message")

    # Example of intercepting logs from another package
    logger.intercept_package_logs("requests")
    import requests

    requests.get("https://example.com")
