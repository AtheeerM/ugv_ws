#!/bin/bash
cd /mnt/ssd/ugv_ws/src/ugv_main/ugv_nav/maps
ros2 run nav2_map_server map_saver_cli -f ./map --ros-args -p map_subscribe_transient_local:=true
cd -
