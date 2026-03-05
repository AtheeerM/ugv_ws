#!/usr/bin/env python3
"""
greedy_4_goals.py
-----------------
Waits for Nav2 to be fully ready before attempting any planning or navigation.
Fixes:
  - do_transform_pose() bypass (broken on some ROS2 Humble builds)
  - Startup delay: waits for both action servers before the loop runs
  - TF extrapolation: retries until a stable transform is available
"""
import math
import time
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.action import ComputePathToPose

import tf2_ros


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def path_length(path) -> float:
    poses = path.poses
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


def pose_from_transform(tf_stamped, frame_id: str) -> PoseStamped:
    """
    Build a PoseStamped directly from a TransformStamped.
    Avoids do_transform_pose() which is broken on some ROS2 Humble builds.
    The robot is at the origin of its own base frame, so its map-frame pose
    IS the transform itself.
    """
    t  = tf_stamped.transform
    ps = PoseStamped()
    ps.header.frame_id    = frame_id
    ps.header.stamp       = tf_stamped.header.stamp
    ps.pose.position.x    = t.translation.x
    ps.pose.position.y    = t.translation.y
    ps.pose.position.z    = t.translation.z
    ps.pose.orientation.x = t.rotation.x
    ps.pose.orientation.y = t.rotation.y
    ps.pose.orientation.z = t.rotation.z
    ps.pose.orientation.w = t.rotation.w
    return ps


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Greedy4Goals(Node):

    # How long to wait for Nav2 servers at startup (seconds)
    NAV2_WAIT_SEC  = 30.0
    # How long to wait for a stable TF at startup (seconds)
    TF_WAIT_SEC    = 20.0
    # Seconds between loop ticks once running
    LOOP_PERIOD    = 2.0

    def __init__(self):
        super().__init__("greedy_4_goals")

        self.map_frame  = "map"
        self.base_frame = "base_footprint"

        self.nav_client  = ActionClient(self, NavigateToPose,    "navigate_to_pose")
        self.plan_client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.plan_timeout_sec = 5.0

        # Your 4 goals (coordinates clicked in RViz)
        self.goals = [
            self.make_goal(-0.84294593334198,   1.9491890668869019, 0.0),
            self.make_goal(-3.521780252456665,  0.4063647389411926, 0.0),
            self.make_goal( 0.4273202121257782, 2.102668523788452,  0.0),
            self.make_goal(-3.9292821884155273, 5.361337184906006,  0.0),
        ]

        self.running  = False
        self.ready    = False   # set True after Nav2 + TF are confirmed ready
        self.timer    = self.create_timer(self.LOOP_PERIOD, self.loop)

        self.get_logger().info("Greedy4Goals started — waiting for Nav2 + TF to be ready...")

        # Kick off async readiness check
        self.create_timer(0.5, self._check_ready_once)

    # ------------------------------------------------------------------
    # Startup readiness gate
    # ------------------------------------------------------------------

    def _check_ready_once(self) -> None:
        """
        Called once shortly after startup.
        Blocks (with spin) until both Nav2 action servers are up and
        a stable TF transform is available.
        After this, self.ready = True and the main loop is unblocked.
        """
        # --- Wait for Nav2 action servers ---
        self.get_logger().info(
            f"Waiting for 'navigate_to_pose' (up to {self.NAV2_WAIT_SEC}s)..."
        )
        if not self.nav_client.wait_for_server(timeout_sec=self.NAV2_WAIT_SEC):
            self.get_logger().error(
                "navigate_to_pose not available after timeout. "
                "Is Nav2 running? Exiting."
            )
            raise SystemExit(1)

        self.get_logger().info(
            f"Waiting for 'compute_path_to_pose' (up to {self.NAV2_WAIT_SEC}s)..."
        )
        if not self.plan_client.wait_for_server(timeout_sec=self.NAV2_WAIT_SEC):
            self.get_logger().error(
                "compute_path_to_pose not available after timeout. "
                "Is Nav2 running? Exiting."
            )
            raise SystemExit(1)

        self.get_logger().info("Nav2 servers ready ✅")

        # --- Wait for a stable TF transform ---
        self.get_logger().info(
            f"Waiting for stable TF ({self.map_frame} → {self.base_frame})..."
        )
        deadline = time.time() + self.TF_WAIT_SEC
        while time.time() < deadline:
            try:
                self.tf_buffer.lookup_transform(
                    self.map_frame, self.base_frame, rclpy.time.Time()
                )
                self.get_logger().info("TF transform available ✅")
                break
            except Exception:
                time.sleep(0.5)
        else:
            self.get_logger().warn(
                f"TF not stable after {self.TF_WAIT_SEC}s — "
                "make sure you set the 2D Pose Estimate in RViz! "
                "Will keep trying in main loop..."
            )

        self.ready = True
        self.get_logger().info(
            f"Ready! Starting greedy navigation over {len(self.goals)} goals."
        )

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------

    def make_goal(self, x: float, y: float, yaw: float) -> PoseStamped:
        g = PoseStamped()
        g.header.frame_id = self.map_frame
        g.pose.position.x = x
        g.pose.position.y = y
        z, w = yaw_to_quat(yaw)
        g.pose.orientation.z = z
        g.pose.orientation.w = w
        return g

    def get_robot_pose_in_map(self) -> PoseStamped | None:
        """Read robot pose from TF directly — no do_transform_pose()."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
            return pose_from_transform(tf, self.map_frame)
        except tf2_ros.ExtrapolationException:
            # Normal at startup — just wait
            return None
        except Exception as e:
            self.get_logger().warn(f"TF error: {e}")
            return None

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def compute_path_cost(
        self, start_map: PoseStamped, goal_map: PoseStamped
    ) -> float | None:

        now = self.get_clock().now().to_msg()
        start_map.header.stamp = now
        goal_map.header.stamp  = now

        plan_goal           = ComputePathToPose.Goal()
        plan_goal.start     = start_map
        plan_goal.goal      = goal_map
        plan_goal.use_start = True

        send_future = self.plan_client.send_goal_async(plan_goal)
        rclpy.spin_until_future_complete(
            self, send_future, timeout_sec=self.plan_timeout_sec
        )
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().debug("Plan goal rejected by server.")
            return None

        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(
            self, res_future, timeout_sec=self.plan_timeout_sec
        )
        res = res_future.result()
        if res is None:
            self.get_logger().debug("Plan result was None (timeout?).")
            return None

        length = path_length(res.result.path)
        if length == 0.0:
            self.get_logger().debug("Planner returned zero-length path — skipping goal.")
            return None

        return length

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate_to(self, goal_map: PoseStamped) -> bool:
        goal_map.header.stamp = self.get_clock().now().to_msg()

        g      = NavigateToPose.Goal()
        g.pose = goal_map

        send_future = self.nav_client.send_goal_async(g)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn("Navigation goal rejected.")
            return False

        self.get_logger().info("Goal accepted, driving...")
        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        res = res_future.result()
        if res is None:
            return False

        from action_msgs.msg import GoalStatus
        return res.status == GoalStatus.STATUS_SUCCEEDED

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def loop(self):
        # Block until Nav2 + TF confirmed ready
        if not self.ready:
            return

        if self.running:
            return

        if not self.goals:
            self.get_logger().info("All goals completed ✅  Shutting down loop.")
            self.timer.cancel()
            return

        # Get robot pose
        start = self.get_robot_pose_in_map()
        if start is None:
            self.get_logger().warn(
                "TF not ready yet — set 2D Pose Estimate in RViz if you haven't!"
            )
            return

        self.running = True

        # --- Score all goals ---
        self.get_logger().info(f"Planning paths to {len(self.goals)} remaining goals...")
        scored = []
        for i, g in enumerate(self.goals):
            cost = self.compute_path_cost(start, g)
            if cost is not None:
                scored.append((cost, i))
                self.get_logger().info(f"  Goal #{i} → {cost:.2f} m")
            else:
                self.get_logger().warn(f"  Goal #{i} → no valid path")

        if not scored:
            self.get_logger().warn(
                "Planner returned no valid paths. Possible causes:\n"
                "  1. Costmap not yet built — wait a few more seconds\n"
                "  2. Goals are inside walls — re-click them in RViz\n"
                "  3. 2D Pose Estimate not set in RViz\n"
                "Retrying in 5s..."
            )
            self.running = False
            return

        # --- Greedy: pick nearest ---
        scored.sort(key=lambda x: x[0])
        best_cost, best_idx = scored[0]
        goal = self.goals.pop(best_idx)

        self.get_logger().info(
            f"→ Navigating to goal #{best_idx} (cost ≈ {best_cost:.2f} m) | "
            f"{len(self.goals)} goals remaining after this"
        )

        ok = self.navigate_to(goal)

        if ok:
            self.get_logger().info("Reached ✅")
        else:
            self.get_logger().warn(
                "Navigation failed ❌ — goal dropped, moving to next."
            )

        self.running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = Greedy4Goals()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
