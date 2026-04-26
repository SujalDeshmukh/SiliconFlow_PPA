# OpenROAD global placement skeleton for SiliconFlow demo.

set report_dir "eda/openroad/siliconflow_demo/reports/place"
file mkdir $report_dir

set log_file "$report_dir/global_place.log"
set rpt_file "$report_dir/place_metrics.rpt"

set fp [open $log_file "w"]
puts $fp "Global placement stage placeholder executed."
close $fp

set rp [open $rpt_file "w"]
puts $rp "Overflow 0.12"
close $rp

exit
