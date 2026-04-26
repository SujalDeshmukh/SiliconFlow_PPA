from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from eda_flow.adapters.openroad_runner import OpenROADRunner, OpenROADStage
from eda_flow.models import (
    EDAEvaluation,
    EDAScoreBreakdown,
    EDAScoreWeights,
    score_evaluation,
    write_evaluation_json,
)
from eda_flow.parsers.openroad_reports import OpenROADReportParser


@dataclass
class OrchestratorConfig:
    design_name: str
    workspace: Path
    outputs_dir: Path
    timeout_sec: int = 1800
    stop_on_fail: bool = True


class EDAOrchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.runner = OpenROADRunner(
            workspace=config.workspace,
            logs_dir=config.outputs_dir / "logs",
            timeout_sec=config.timeout_sec,
        )
        self.parser = OpenROADReportParser()

    def run(self, stages: List[OpenROADStage]) -> tuple[EDAEvaluation, EDAScoreBreakdown]:
        run_id = uuid4().hex[:10]
        started = datetime.now(timezone.utc)
        stage_results = self.runner.run_flow(stages, stop_on_fail=self.config.stop_on_fail)
        finished = datetime.now(timezone.utc)

        report_paths: List[Path] = []
        for stage in stage_results:
            report_paths.extend(Path(p) for p in stage.report_files)
            report_paths.append(Path(stage.log_file))

        metrics = self.parser.parse(report_paths)
        evaluation = EDAEvaluation.new(
            run_id=run_id,
            design_name=self.config.design_name,
            metrics=metrics,
            stage_results=stage_results,
            reports_parsed=[str(p) for p in report_paths if p.exists()],
            started_at=started,
            finished_at=finished,
        )
        score = score_evaluation(metrics, EDAScoreWeights())
        return evaluation, score

    def persist(self, evaluation: EDAEvaluation, score: EDAScoreBreakdown) -> Dict[str, str]:
        run_dir = self.config.outputs_dir / evaluation.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        eval_file = run_dir / "evaluation.json"
        score_file = run_dir / "score.json"
        write_evaluation_json(eval_file, evaluation)
        score_file.write_text(score.model_dump_json(indent=2), encoding="utf-8")
        return {
            "run_dir": str(run_dir.resolve()),
            "evaluation_file": str(eval_file.resolve()),
            "score_file": str(score_file.resolve()),
        }
