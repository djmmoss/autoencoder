############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 2015 Xilinx Inc. All rights reserved.
############################################################
open_project -reset auto_enc_prj
set_top auto_enc
add_files auto_enc.cpp
add_files -tb auto_enc_test.cpp
add_files -tb data
open_solution -reset "solution1"
set_part {xc7k410tffg900-2}
create_clock -period 5 -name default
#csim_design
csynth_design
#cosim_design
# export_design -format ip_catalog
exit
