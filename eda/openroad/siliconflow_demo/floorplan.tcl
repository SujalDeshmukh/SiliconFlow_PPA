# OpenROAD floorplan skeleton for SiliconFlow demo.
# Replace placeholders with your real LEF/DEF/liberty/netlist inputs.

set report_dir "eda/openroad/siliconflow_demo/reports/floorplan"
file mkdir $report_dir

set log_file "$report_dir/floorplan.log"
set rpt_file "$report_dir/floorplan_metrics.rpt"

set fp [open $log_file "w"]
puts $fp "Floorplan stage placeholder executed."
close $fp

set rp [open $rpt_file "w"]
puts $rp "Area 10000.0"
close $rp

exit
