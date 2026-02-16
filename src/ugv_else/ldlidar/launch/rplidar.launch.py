#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

'''
Parameter Description:
---
- Set laser scan directon:
  1. Set counterclockwise, example: {'laser_scan_dir': True}
  2. Set clockwise,        example: {'laser_scan_dir': False}
- Angle crop setting, Mask data within the set angle range:
  1. Enable angle crop fuction:
    1.1. enable angle crop,  example: {'enable_angle_crop_func': True}
    1.2. disable angle crop, example: {'enable_angle_crop_func': False}
  2. Angle cropping interval setting:
  - The distance and intensity data within the set angle range will be set to 0.
  - angle >= 'angle_crop_min' and angle <= 'angle_crop_max' which is [angle_crop_min, angle_crop_max], unit is degress.
    example:
      {'angle_crop_min': 135.0}
      {'angle_crop_max': 225.0}
      which is [135.0, 225.0], angle unit is degress.
'''

def generate_launch_description():
  # RPLIDAR publisher node (keep names/structure the same as your template)
  rplidar_node = Node(
      package='rplidar_ros',
      executable='rplidar_composition',
      name='rplidar',               
      output='screen',
      parameters=[
        {'frame_id': 'base_lidar_link'},     # keep SAME as your template
        {'serial_port': '/dev/ttyACM0'},     # keep SAME port path as your template
        {'serial_baudrate': 230400},         # keep SAME baudrate as your template

        # "ranges and specifications" (your rplidar launch settings)
        {'angle_compensate': True},
        {'scan_mode': 'Standard'}
      ]
  )

  # base_link to base_laser tf node (keep SAME as your template)
  base_footprint_to_laser_tf_node = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='base_footprint_to_base_laser_rplidar',   # keep SAME as your template
    arguments=['0','0','0','0','0','0','base_footprint','base_lidar_link']
  )

  # Define LaunchDescription variable (keep SAME)
  rp = LaunchDescription()

  rp.add_action(rplidar_node)
  # ld.add_action(base_footprint_to_laser_tf_node)

  return rp