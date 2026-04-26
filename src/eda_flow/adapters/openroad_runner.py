from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from eda_flow.models import EDAStageResult


@dataclass
class OpenROADStage:
    name: str
    command: Sequence[str]
    report_globs: Sequence[str]


class OpenROADRunner:
    def __init__(
        self,
        workspace: Path,
        logs_dir: Path,
        timeout_sec: int = 1800,
    ):
        self.workspace = workspace
        self.logs_dir = logs_dir
        self.timeout_sec = timeout_sec

    def run_stage(self, stage: OpenROADStage) -> EDAStageResult:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"{stage.name}.log"
        started = time.perf_counter()

        with log_file.open("w", encoding="utf-8") as log_fp:
            log_fp.write(f"$ {' '.join(stage.command)}\n\n")
            try:
                proc = subprocess.run(
                    stage.command,
                    cwd=self.workspace,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout_sec,
                    check=False,
                    env=os.environ.copy(),
                    text=True,
                )
                return_code = proc.returncode
                success = return_code == 0
                errors: List[str] = []
            except subprocess.TimeoutExpired:
                return_code = -1
                success = False
                errors = [f"Stage '{stage.name}' timed out after {self.timeout_sec}s"]
            except FileNotFoundError:
                return_code = -2
                success = False
                errors = [
                    f"Executable not found for stage '{stage.name}': {stage.command[0]}"
                ]

        elapsed = time.perf_counter() - started
        report_files: List[str] = []
        for glob_pattern in stage.report_globs:
            for path in self.workspace.glob(glob_pattern):
                if path.is_file():
                    report_files.append(str(path.resolve()))

        return EDAStageResult(
            stage_name=stage.name,
            success=success,
            return_code=return_code,
            command=" ".join(stage.command),
            elapsed_sec=round(elapsed, 4),
            log_file=str(log_file.resolve()),
            report_files=sorted(set(report_files)),
            errors=errors,
        )

    def run_flow(self, stages: Sequence[OpenROADStage], stop_on_fail: bool = True) -> List[EDAStageResult]:
        results: List[EDAStageResult] = []
        for stage in stages:
            result = self.run_stage(stage)
            results.append(result)
            if stop_on_fail and not result.success:
                break
        return results
