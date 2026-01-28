import logging
import os
from datetime import datetime
from typing import Optional
import sys
from exception import CustomException

LOG_DIR = os.path.join(os.getcwd(), "logs")


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _log_file_path() -> str:
    _ensure_log_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(LOG_DIR, f"run_{timestamp}.log")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a configured logger instance.

    - Writes to console and to a timestamped log file in ./logs/
    - Prevents duplicate handlers if called multiple times
    - Use: logger = get_logger(__name__)
    """
    logger_name = name if name else "student_performance"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times (very common in notebooks/servers)
    if logger.handlers:
        return logger

    log_path = _log_file_path()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] Line %(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Avoid double logging through root logger
    logger.propagate = False

    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger


# Convenience default logger (optional)
logging = get_logger("student_performance")