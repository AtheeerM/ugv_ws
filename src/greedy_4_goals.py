#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ComputePathToPose

import tf2_ros
from tf2_geometry_msgs import do_transform_pose


def path_length(nav_path) -> float:
    poses = nav_path.poses
    if len(poses) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(poses)):
        x0 = poses[i - 1].pose.position.x
        y0 = poses[i - 1].pose.position.y
        x1 = poses[i].pose.position.x
        y1 = poses[i].pose.position.y
        dist += math.hypot(x1 - x0, y1 - y0)
    return dist


def yaw_to_quat(yaw: float):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class Greedy4Goals(Node):
    def __init__(self):
        super().__init__("greedy_4_goals")

        # Nav2 default interfaces (we can change if your names differ)
        self.nav_action_name = "navigate_to_pose"
        self.plan_service_name = "compute_path_to_pose"

        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action_name)
        self.plan_client = self.create_client(ComputePathToPose, self.plan_service_name)

        # TF to get robot pose in map
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # TODO: we will replace these 4 placeholders with your RViz points next
        self.goals = [
            self.make_goal(0.0, 0.0, 0.0),
            self.make_goal(0.0, 0.0, 0.0),
            self.make_goal(0.0, 0.0, 0.0),
            self.make_goal(0.0, 0.0, 0.0),
        ]

        self.running = False
        self.timer = self.create_timer(1.0, self.loop)
        self.get_logger().info("Greedy4Goals started. Initialize pose in RViz, then run this node.")

    def make_goal(self, x: float, y: float, yaw: float) -> PoseStamped:
        g = PoseStamped()
        g.header.frame_id = "map"
        g.pose.position.x = x
        g.pose.position.y = y
        z, w = yaw_to_quat(yaw)
        g.pose.orientation.z = z
        g.pose.orientation.w = w
        return g

    def get_robot_pose_in_map(self) -> PoseStamped | None:
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            p = PoseStamped()
            p.header.frame_id = "base_link"
            p.pose.orientation.w = 1.0
            return do_transform_pose(p, tf)
        except Exception as e:
            self.get_logger().warn(f"TF not ready (map->base_link): {e}")
            return None

    def compute_path_cost(self, start: PoseStamped, goal: PoseStamped) -> float | None:
        if not self.plan_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(f"Service '{self.plan_service_name}' not available yet")
            return None

        req = ComputePathToPose.Request()
        req.start = start
        req.goal = goal
        req.use_start = True

        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.result() is None:
            return None

        return path_length(future.result().path)

    def navigate_to(self, goal: PoseStamped) -> bool:
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f"Action '{self.nav_action_name}' not available")
            return False

        msg = NavigateToPose.Goal()
        msg.pose = goal

        send_future = self.nav_client.send_goal_async(msg)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=2.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        status = result_future.result().status

        # action_msgs/GoalStatus: 4 = SUCCEEDED
        return status == 4

    def loop(self):
        if self.running:
            return
        if len(self.goals) == 0:
            self.get_logger().info("All 4 destinations completed ✅")
            self.timer.cancel()
            return

        start = self.get_robot_pose_in_map()
        if start is None:
            return

        self.running = True

        best_i = None
        best_cost = None
        for i, g in enumerate(self.goals):
            cost = self.compute_path_cost(start, g)
            if cost is None:
                continue
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_i = i

        if best_i is None:
            self.get_logger().warn("Could not compute a path to any goal. Check Nav2 is running.")
            self.running = False
            return

        next_goal = self.goals.pop(best_i)
        self.get_logger().info(f"Chosen next goal automatically (cost ~ {best_cost:.2f} m).")

        ok = self.navigate_to(next_goal)
        self.get_logger().info("Reached ✅" if ok else "Failed/aborted ❌")

        self.running = False


def main():
    rclpy.init()
    node = Greedy4Goals()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
