#!/usr/bin/env python3
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial


class LoraReader(Node):

    def __init__(self):
        super().__init__('lora_reader')
        self.publisher = self.create_publisher(
            String, '/lora_tag', 10
        )
        try:
            self.serial_port = serial.Serial(
                '/dev/ttyUSB1', 115200, timeout=1.0
            )
            self.get_logger().info(
                "LoRa Reader started on /dev/ttyUSB1"
            )
        except Exception as e:
            self.get_logger().error(f"Serial port error: {e}")
            return

        self.thread = threading.Thread(
            target=self.read_serial, daemon=True
        )
        self.thread.start()

    def read_serial(self):
        while rclpy.ok():
            try:
                if not hasattr(self, 'serial_port'):
                    time.sleep(0.1)
                    continue

                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline()
                    line = line.decode('utf-8', errors='ignore').strip()

                    # Accept "TAG_001" or "TAG_001,-73"
                    if line.startswith("TAG_"):
                        parts  = line.split(',')
                        tag_id = parts[0].strip()
                        rssi   = parts[1].strip() \
                            if len(parts) > 1 and parts[1].strip() \
                            else "0"

                        if tag_id in ["TAG_001", "TAG_002",
                                      "TAG_003", "TAG_004"]:
                            # Publish as "TAG_001,-73"
                            msg      = String()
                            msg.data = f"{tag_id},{rssi}"
                            self.publisher.publish(msg)
                            self.get_logger().info(
                                f"Published: {msg.data}"
                            )
                else:
                    time.sleep(0.01)

            except Exception as e:
                self.get_logger().warn(f"Serial error: {e}")

    def destroy_node(self):
        if hasattr(self, 'serial_port') and self.serial_port.is_open:
            self.serial_port.close()
            self.get_logger().info("Serial port closed.")
        super().destroy_node()


def main():
    rclpy.init()
    node = LoraReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()