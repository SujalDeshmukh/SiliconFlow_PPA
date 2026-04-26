from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EDAMetrics(BaseModel):
    area_um2: Optional[float] = None
    total_power_mw: Optional[float] = None
    leakage_power_mw: Optional[float] = None
    dynamic_power_mw: Optional[float] = None
    wns_ns: Optional[float] = None
    tns_ns: Optional[float] = None
    congestion_overflow: Optional[float] = None
    drc_violations: Optional[int] = None


class EDAStageResult(BaseModel):
    stage_name: str
    success: bool
    return_code: int
    command: str
    elapsed_sec: float
    log_file: str
    report_files: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class EDAEvaluation(BaseModel):
    run_id: str
    design_name: str
    toolchain: str = "openroad"
    started_at_utc: str
    finished_at_utc: str
    metrics: EDAMetrics
    stage_results: List[EDAStageResult] = Field(default_factory=list)
    reports_parsed: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)

    @classmethod
    def new(
        cls,
        run_id: str,
        design_name: str,
        metrics: EDAMetrics,
        stage_results: List[EDAStageResult],
        reports_parsed: List[str],
        metadata: Optional[Dict[str, str]] = None,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
    ) -> "EDAEvaluation":
        started = started_at or datetime.now(timezone.utc)
        finished = finished_at or datetime.now(timezone.utc)
        return cls(
            run_id=run_id,
            design_name=design_name,
            metrics=metrics,
            stage_results=stage_results,
            reports_parsed=reports_parsed,
            metadata=metadata or {},
            started_at_utc=started.isoformat(),
            finished_at_utc=finished.isoformat(),
        )


class EDAScoreWeights(BaseModel):
    area: float = 1.0
    power: float = 1.0
    timing: float = 2.0
    congestion: float = 1.5
    drc: float = 2.0


class EDAScoreBreakdown(BaseModel):
    total_score: float
    area_term: float = 0.0
    power_term: float = 0.0
    timing_term: float = 0.0
    congestion_term: float = 0.0
    drc_term: float = 0.0


def score_evaluation(
    metrics: EDAMetrics,
    weights: Optional[EDAScoreWeights] = None,
) -> EDAScoreBreakdown:
    w = weights or EDAScoreWeights()

    area_term = -w.area * (metrics.area_um2 or 0.0)
    power_term = -w.power * (metrics.total_power_mw or 0.0)

    wns = metrics.wns_ns if metrics.wns_ns is not None else -1.0
    tns = metrics.tns_ns if metrics.tns_ns is not None else -10.0
    timing_penalty = max(0.0, -wns) + max(0.0, -tns)
    timing_term = -w.timing * timing_penalty

    congestion_term = -w.congestion * (metrics.congestion_overflow or 0.0)
    drc_term = -w.drc * float(metrics.drc_violations or 0)

    total = area_term + power_term + timing_term + congestion_term + drc_term
    return EDAScoreBreakdown(
        total_score=round(total, 6),
        area_term=round(area_term, 6),
        power_term=round(power_term, 6),
        timing_term=round(timing_term, 6),
        congestion_term=round(congestion_term, 6),
        drc_term=round(drc_term, 6),
    )


def write_evaluation_json(output_file: Path, evaluation: EDAEvaluation) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
