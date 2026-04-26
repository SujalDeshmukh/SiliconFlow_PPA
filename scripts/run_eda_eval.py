"""
Starter EDA evaluation loop for real report integration.

This script runs configured flow stages, parses generated reports/logs,
and emits unified evaluation + score JSON artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(__file__)
SRC = os.path.abspath(os.path.join(HERE, "../src"))
sys.path.insert(0, SRC)

from eda_flow.adapters.openroad_runner import OpenROADStage
from eda_flow.orchestrator import EDAOrchestrator, OrchestratorConfig


def parse_stage(stage_text: str) -> OpenROADStage:
    # Format: name::command::report_glob_1,report_glob_2
    parts = stage_text.split("::")
    if len(parts) != 3:
        raise ValueError(
            "Invalid --stage value. Expected format: "
            "name::command::report_glob_1,report_glob_2"
        )
    name, command, report_globs = parts
    cmd_tokens = [p for p in command.strip().split(" ") if p]
    globs = [g.strip() for g in report_globs.split(",") if g.strip()]
    return OpenROADStage(name=name.strip(), command=cmd_tokens, report_globs=globs)


def default_stages() -> list[OpenROADStage]:
    # Safe defaults so the pipeline can run without OpenROAD installed.
    # Replace with real flow commands when integrating your tool environment.
    return [
        OpenROADStage(
            name="mock_floorplan",
            command=["python", "-c", "print('WNS -0.12\\nTNS -1.54\\nTotal Power 12.8\\nArea 10345.2')"],
            report_globs=[],
        ),
        OpenROADStage(
            name="mock_route",
            command=["python", "-c", "print('Overflow 0.08\\nDRC violations 3')"],
            report_globs=[],
        ),
    ]


def _extract_tcl_path(command: list[str]) -> Path:
    for token in reversed(command):
        if token.lower().endswith(".tcl"):
            return Path(token)
    raise ValueError(
        "OpenROAD stage command must contain a .tcl script path. "
        f"Command: {' '.join(command)}"
    )


def _to_workspace_relative(path_value: Path, workspace: Path) -> str:
    candidate = path_value if path_value.is_absolute() else (workspace / path_value)
    resolved = candidate.resolve()
    try:
        rel = resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Tcl script '{resolved}' is outside workspace '{workspace}'. "
            "Use paths under the mounted workspace."
        ) from exc
    return rel.as_posix()


def dockerize_openroad_stages(
    stages: list[OpenROADStage],
    workspace: Path,
    docker_image: str,
) -> list[OpenROADStage]:
    dockerized: list[OpenROADStage] = []
    volume_arg = f"{str(workspace.resolve())}:/work"
    for stage in stages:
        is_openroad_stage = any(token.lower() == "openroad" for token in stage.command)
        if not is_openroad_stage:
            dockerized.append(stage)
            continue

        tcl_path = _extract_tcl_path(list(stage.command))
        tcl_rel = _to_workspace_relative(tcl_path, workspace)
        # Try user-requested invocation first; if binary not in PATH, fallback path runs.
        container_shell = (
            f"openroad {tcl_rel} || "
            f"/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad {tcl_rel}"
        )
        dockerized.append(
            OpenROADStage(
                name=stage.name,
                command=[
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    volume_arg,
                    "-w",
                    "/work",
                    docker_image,
                    "/bin/bash",
                    "-lc",
                    container_shell,
                ],
                report_globs=stage.report_globs,
            )
        )
    return dockerized


def load_stages_from_config(config_file: Path) -> tuple[dict, list[OpenROADStage]]:
    payload = json.loads(config_file.read_text(encoding="utf-8"))
    stage_defs = payload.get("stages", [])
    stages: list[OpenROADStage] = []
    for stage in stage_defs:
        stages.append(
            OpenROADStage(
                name=stage["name"],
                command=stage["command"],
                report_globs=stage.get("report_globs", []),
            )
        )
    return payload, stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA flow and parse real metrics.")
    parser.add_argument("--design-name", default="demo_design")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--outputs-dir", default="artifacts/eda_runs")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--docker-image", default="openroad/orfs")
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Run stage commands directly without Docker wrapping.",
    )
    parser.add_argument(
        "--stage-config",
        default="",
        help="Path to JSON stage config (see configs/openroad_stages.example.json)",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Stage spec: name::command::report_glob_1,report_glob_2",
    )
    args = parser.parse_args()

    design_name = args.design_name
    workspace = Path(args.workspace).resolve()
    outputs_dir = Path(args.outputs_dir).resolve()
    timeout_sec = args.timeout_sec

    if args.stage_config:
        config_payload, stages = load_stages_from_config(Path(args.stage_config).resolve())
        design_name = config_payload.get("design_name", design_name)
        workspace = Path(config_payload.get("workspace", str(workspace))).resolve()
        outputs_dir = Path(config_payload.get("outputs_dir", str(outputs_dir))).resolve()
        timeout_sec = int(config_payload.get("timeout_sec", timeout_sec))
        docker_image = config_payload.get("docker_image", args.docker_image)
    else:
        stages = [parse_stage(s) for s in args.stage] if args.stage else default_stages()
        docker_image = args.docker_image

    if not args.no_docker:
        stages = dockerize_openroad_stages(stages, workspace=workspace, docker_image=docker_image)

    cfg = OrchestratorConfig(
        design_name=design_name,
        workspace=workspace,
        outputs_dir=outputs_dir,
        timeout_sec=timeout_sec,
    )
    orchestrator = EDAOrchestrator(cfg)
    evaluation, score = orchestrator.run(stages)
    paths = orchestrator.persist(evaluation, score)

    print("EDA evaluation complete.")
    print(f"Run ID: {evaluation.run_id}")
    print(f"Total score: {score.total_score}")
    print(f"Evaluation file: {paths['evaluation_file']}")
    print(f"Score file: {paths['score_file']}")


if __name__ == "__main__":
    main()
