# OpenROAD Setup for SiliconFlow-PPA

This is the fastest path to your first real EDA-coupled run.

## 1) Use the design-specific stage config

Use:

- `configs/openroad_stages.my_design.json`

Edit only these values first:

- `design_name` (replace `my_design`)
- every `command` Tcl path under `eda/openroad/my_design/`
- every `report_globs` path so they match where your Tcl scripts write reports

## 2) Create your flow script layout

Create these files:

- `eda/openroad/my_design/floorplan.tcl`
- `eda/openroad/my_design/global_place.tcl`
- `eda/openroad/my_design/global_route.tcl`
- `eda/openroad/my_design/sta.tcl`

And ensure your scripts generate reports under:

- `eda/openroad/my_design/reports/floorplan/`
- `eda/openroad/my_design/reports/place/`
- `eda/openroad/my_design/reports/route/`
- `eda/openroad/my_design/reports/sta/`

## 3) Run evaluation

```bash
python scripts/run_eda_eval.py --stage-config configs/openroad_stages.my_design.json
```

Outputs:

- `artifacts/eda_runs/<run_id>/evaluation.json`
- `artifacts/eda_runs/<run_id>/score.json`

## 4) Validate parsed metrics

`evaluation.json` should contain at least some of:

- `area_um2`
- `total_power_mw`
- `wns_ns`
- `tns_ns`
- `congestion_overflow`
- `drc_violations`

If values are missing, adjust report file patterns and/or report text format in Tcl outputs.
