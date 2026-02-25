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
    # planar yaw -> quaternion (z,w), x=y=0
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class Greedy4Goals(Node):
    def __init__(self):
        super().__init__("greedy_4_goals")

        # Nav2 interfaces
        self.nav_action_name = "navigate_to_pose"
        self.plan_service_name = "compute_path_to_pose"

        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action_name)
        self.plan_client = self.create_client(ComputePathToPose, self.plan_service_name)

        # TF to get robot pose in map
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- TUNABLES ---
        self.max_goal_attempts_per_cycle = 4   # try up to N best goals each cycle (usually == len(goals))
        self.retry_delay_sec = 2.0             # if everything fails, wait before trying again
        self.plan_timeout_sec = 2.0
        self.action_server_wait_sec = 2.0

        # Replace these with YOUR 4 points
        self.goals = [
            self.make_goal(1.4691200256347656, -1.1307926177978516, 0.0),
            self.make_goal(1.905529260635376, -0.2111365646123886, 0.0),
            self.make_goal(0.4719597101211548,  2.088629722595215, 0.0),
            self.make_goal(-1.3124245405197144, -0.2761569023132324, 0.0),
        ]

        self.running = False
        self.timer = self.create_timer(1.0, self.loop)
        self.get_logger().info("Greedy4Goals started. Make sure Nav2 is running + pose is initialized in RViz.")

    def make_goal(self, x: float, y: float, yaw: float) -> PoseStamped:
        g = PoseStamped()
        g.header.frame_id = "map"
        # IMPORTANT: timestamp helps TF / Nav2 timing
        g.header.stamp = self.get_clock().now().to_msg()
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
            p.header.stamp = self.get_clock().now().to_msg()
            p.pose.orientation.w = 1.0
            return do_transform_pose(p, tf)
        except Exception as e:
            self.get_logger().warn(f"TF not ready (map->base_link): {e}")
            return None

    def compute_path_cost(self, start: PoseStamped, goal: PoseStamped) -> float | None:
        if not self.plan_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(f"Service '{self.plan_service_name}' not available yet")
            return None

        # Stamp both start and goal (Nav2 likes fresh stamps)
        start.header.stamp = self.get_clock().now().to_msg()
        goal.header.stamp = self.get_clock().now().to_msg()

        req = ComputePathToPose.Request()
        req.start = start
        req.goal = goal
        req.use_start = True

        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.plan_timeout_sec)
        if future.result() is None:
            return None

        return path_length(future.result().path)

    def navigate_to(self, goal: PoseStamped) -> tuple[bool, int]:
        """
        Returns (success, status_code)
        status_code is action_msgs/GoalStatus value.
        """
        if not self.nav_client.wait_for_server(timeout_sec=self.action_server_wait_sec):
            self.get_logger().error(f"Action '{self.nav_action_name}' not available")
            return False, -1

        goal.header.stamp = self.get_clock().now().to_msg()

        msg = NavigateToPose.Goal()
        msg.pose = goal

        send_future = self.nav_client.send_goal_async(msg)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=2.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Goal rejected")
            return False, -2

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            return False, -3

        status = result.status
        # action_msgs/GoalStatus: 4 = SUCCEEDED
        return (status == 4), status

    def loop(self):
        if self.running:
            return

        if len(self.goals) == 0:
            self.get_logger().info("All destinations completed ✅")
            self.timer.cancel()
            return

        start = self.get_robot_pose_in_map()
        if start is None:
            return

        self.running = True

        # 1) compute cost to every remaining goal
        scored = []
        for i, g in enumerate(self.goals):
            cost = self.compute_path_cost(start, g)
            if cost is None:
                continue
            scored.append((cost, i))

        if not scored:
            self.get_logger().warn("No valid paths to any goal right now. Will retry soon...")
            self.running = False
            return

        # 2) sort by best cost (greedy)
        scored.sort(key=lambda x: x[0])

        # 3) Try goals from best to worse until one succeeds
        attempts = 0
        success_index = None

        for cost, idx in scored:
            attempts += 1
            if attempts > self.max_goal_attempts_per_cycle:
                break

            g = self.goals[idx]
            self.get_logger().info(f"Trying goal #{idx} (greedy cost ~ {cost:.2f} m) ...")

            ok, status = self.navigate_to(g)

            if ok:
                self.get_logger().info("Reached ✅")
                success_index = idx
                break
            else:
                self.get_logger().warn(f"Failed/aborted ❌ (status={status}). Trying next best goal...")

        # 4) Only remove goal if reached
        if success_index is not None:
            self.goals.pop(success_index)
            self.get_logger().info(f"Remaining goals: {len(self.goals)}")
        else:
            self.get_logger().warn(f"All attempted goals failed this cycle. Waiting {self.retry_delay_sec}s then retry...")
            # simple delay before next cycle
            # (keeps it easy; if your system is slow this helps stabilize)
            self.running = False
            self.create_timer(self.retry_delay_sec, self._unlock_once)
            return

        self.running = False

    def _unlock_once(self):
        # called once by the retry timer to re-enable loop
        # (timer auto-cancels by not rescheduling itself)
        pass


def main():
    rclpy.init()
    node = Greedy4Goals()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
