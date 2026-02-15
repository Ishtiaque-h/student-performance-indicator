import logging as py_logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def _find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return start


def _log_dir() -> Path:
    return _find_project_root() / "logs"


def _log_file_path() -> Path:
    _log_dir().mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return _log_dir() / f"run_{timestamp}.log"


def get_logger(name: Optional[str] = None) -> py_logging.Logger:
    """
    Returns a configured logger instance.

    - Writes to console and to a timestamped log file in ./logs/
    - Prevents duplicate handlers if called multiple times
    - Use: logger = get_logger(__name__)
    """
    logger_name = name if name else "student_performance"
    logger = py_logging.getLogger(logger_name)
    logger.setLevel(py_logging.INFO)

    # Prevent adding handlers multiple times (very common in notebooks/servers)
    if logger.handlers:
        return logger

    log_path = _log_file_path()

    formatter = py_logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] Line %(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = py_logging.StreamHandler()
    console_handler.setLevel(py_logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = py_logging.FileHandler(str(log_path))
    file_handler.setLevel(py_logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Avoid double logging through root logger
    logger.propagate = False

    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger


# Convenience default logger
logging = get_logger("student_performance")
