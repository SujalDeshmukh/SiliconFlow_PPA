from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_value: str | None, default_relative: str) -> Path:
    if path_value:
        path = Path(path_value)
        return path if path.is_absolute() else _repo_root() / path
    return _repo_root() / default_relative


def load_json(path_value: str | None, default_relative: str) -> Dict[str, Any]:
    path = _resolve_path(path_value, default_relative)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_runtime_setting(name: str, default: str) -> str:
    return os.environ.get(name, default)

