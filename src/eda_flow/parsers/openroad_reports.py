from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from eda_flow.models import EDAMetrics
from eda_flow.parsers.base import EDAReportParser


_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _extract_first_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (ValueError, IndexError):
        return None


def _extract_first_int(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (ValueError, IndexError):
        return None


class OpenROADReportParser(EDAReportParser):
    """
    Parse common OpenROAD report snippets from text files.
    Supports mixed reports/logs and extracts the first useful values.
    """

    def parse(self, report_files: Iterable[Path]) -> EDAMetrics:
        metrics = EDAMetrics()
        for report_file in report_files:
            if not report_file.exists() or not report_file.is_file():
                continue

            text = report_file.read_text(encoding="utf-8", errors="ignore")
            metrics = self._merge_metrics(metrics, self._parse_single(text))
        return metrics

    def _parse_single(self, text: str) -> EDAMetrics:
        area = _extract_first_float(rf"(?:core\s+)?area[^0-9\-+]*({_FLOAT})", text)
        total_power = _extract_first_float(rf"total\s+power[^0-9\-+]*({_FLOAT})", text)
        leakage_power = _extract_first_float(rf"leakage(?:\s+power)?[^0-9\-+]*({_FLOAT})", text)
        dynamic_power = _extract_first_float(rf"dynamic(?:\s+power)?[^0-9\-+]*({_FLOAT})", text)

        wns = _extract_first_float(rf"\bWNS\b[^0-9\-+]*({_FLOAT})", text)
        tns = _extract_first_float(rf"\bTNS\b[^0-9\-+]*({_FLOAT})", text)

        congestion = _extract_first_float(rf"(?:total\s+)?overflow[^0-9\-+]*({_FLOAT})", text)
        drc = _extract_first_int(r"DRC[^0-9]*(\d+)", text)
        if drc is None:
            drc = _extract_first_int(r"(?:violations?|vio)[^0-9]*(\d+)", text)

        return EDAMetrics(
            area_um2=area,
            total_power_mw=total_power,
            leakage_power_mw=leakage_power,
            dynamic_power_mw=dynamic_power,
            wns_ns=wns,
            tns_ns=tns,
            congestion_overflow=congestion,
            drc_violations=drc,
        )

    @staticmethod
    def _merge_metrics(base: EDAMetrics, update: EDAMetrics) -> EDAMetrics:
        base_data = base.model_dump()
        update_data = update.model_dump()
        for key, value in update_data.items():
            if value is not None:
                base_data[key] = value
        return EDAMetrics(**base_data)
