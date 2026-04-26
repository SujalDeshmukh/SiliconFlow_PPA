from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from eda_flow.models import EDAMetrics


class EDAReportParser(ABC):
    @abstractmethod
    def parse(self, report_files: Iterable[Path]) -> EDAMetrics:
        raise NotImplementedError
