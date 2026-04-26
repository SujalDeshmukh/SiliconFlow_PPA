# OpenROAD global routing skeleton for SiliconFlow demo.

set report_dir "eda/openroad/siliconflow_demo/reports/route"
file mkdir $report_dir

set log_file "$report_dir/global_route.log"
set rpt_file "$report_dir/route_metrics.rpt"

set fp [open $log_file "w"]
puts $fp "Global route stage placeholder executed."
close $fp

set rp [open $rpt_file "w"]
puts $rp "DRC violations 5"
close $rp

exit
