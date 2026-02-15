from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import dill

from student_performance.exception import CustomException
from student_performance.logger import logging


def find_project_root(
    start: Path | None = None, markers: Sequence[str] = ("pyproject.toml", ".git")
) -> Path:
    """
    Find repo/project root by walking up from start until a marker is found.
    Default markers: pyproject.toml or .git
    """
    start = (start or Path.cwd()).resolve()
    for p in (start, *start.parents):
        for m in markers:
            if (p / m).exists():
                return p
    return start


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure the parent directory exists for a file path OR ensure directory exists for a dir path.
    Returns the resolved Path.
    """
    p = Path(path).expanduser().resolve()
    # If it has a suffix, assume it's a file path; otherwise it's a dir path
    dir_path = p.parent if p.suffix else p
    dir_path.mkdir(parents=True, exist_ok=True)
    return p


def save_object(file_path: str | Path, obj: Any) -> None:
    """
    Serialize and save a Python object to disk using dill.
    Uses an atomic write: write temp file then replace final artifact.
    """
    try:
        file_path = ensure_dir(file_path)

        # Write to temp file in same directory, then replace (atomic on most OS/filesystems)
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(file_path.parent),
            prefix=file_path.name + ".",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)

        try:
            with os.fdopen(tmp_fd, "wb") as f:
                dill.dump(obj, f)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(file_path)
        finally:
            # If something failed before replace, clean temp file
            if tmp_path.exists() and tmp_path != file_path:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        logging.error(f"Failed to save object at: {file_path}")
        raise CustomException(e, sys)


def load_object(file_path: str | Path) -> Any:
    """
    Load a serialized Python object from disk using dill.
    NOTE: Do not load untrusted dill/pickle files (can execute code).
    """
    try:
        file_path = Path(file_path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found: {file_path}")

        with open(file_path, "rb") as f:
            return dill.load(f)

    except Exception as e:
        logging.error(f"Failed to load object from: {file_path}")
        raise CustomException(e, sys)
