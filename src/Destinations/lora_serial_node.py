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
                '/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-port0',
                115200, timeout=1.0
            )
            self.get_logger().info("LoRa Reader started successfully")
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

                    if "TAG_" in line:
                        # Extract tag ID
                        tag_start = line.index("TAG_")
                        tag_id = line[tag_start:tag_start+7]  # e.g. "TAG_001"

                        # Extract RSSI
                        rssi = "0"
                        if "RSSI:" in line:
                            rssi = line.split("RSSI:")[-1].strip()  # e.g. "-70"

                        if tag_id in ["TAG_001", "TAG_002",
                                      "TAG_003", "TAG_004"]:
                            msg = String()
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