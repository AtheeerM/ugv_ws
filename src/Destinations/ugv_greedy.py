#!/usr/bin/env python3
import math, time, threading, subprocess, rclpy, rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose, ComputePathToPose
from action_msgs.msg import GoalStatus
import tf2_ros


def yaw_to_quat(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def pose_from_tf(tf_stamped, frame_id):
    t = tf_stamped.transform
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.header.stamp = tf_stamped.header.stamp
    ps.pose.position.x = t.translation.x
    ps.pose.position.y = t.translation.y
    ps.pose.position.z = t.translation.z
    ps.pose.orientation.x = t.rotation.x
    ps.pose.orientation.y = t.rotation.y
    ps.pose.orientation.z = t.rotation.z
    ps.pose.orientation.w = t.rotation.w
    return ps


class Greedy4Goals(Node):

    OBSTACLE_REPLAN_RADIUS = 2.0  # meters — use path cost only when goal is this close

    def __init__(self):
        super().__init__("greedy_4_goals")
        self.map_frame  = "map"
        self.base_frame = "base_footprint"
        self.cb_group = ReentrantCallbackGroup()

        self.nav_client  = ActionClient(self, NavigateToPose,    "navigate_to_pose",    callback_group=self.cb_group)
        self.plan_client = ActionClient(self, ComputePathToPose, "compute_path_to_pose", callback_group=self.cb_group)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # FIX: Twist publisher must be created in __init__, not inside stop_robot
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        self.goals = [
            self.make_goal(3.99,   0.0213,   0.00205),
            self.make_goal(2.99,   0.984,    0.00026),
            self.make_goal(0.991,  1.99,     0.00126),
            self.make_goal(2.02,  -0.00146,  0.00243),
        ]

        self.get_logger().info("Waiting for Nav2 action server...")
        self.nav_client.wait_for_server()
        self.plan_client.wait_for_server()
        self.get_logger().info("Nav2 ready! Waiting 15s for full initialization...")
        time.sleep(15.0)
        self.get_logger().info("Starting greedy navigation!")

        self._nav_thread = threading.Thread(target=self.navigation_loop, daemon=True)
        self._nav_thread.start()

    # ------------------------------------------------------------------
    # Stop robot immediately
    # ------------------------------------------------------------------

    def stop_robot(self):
        # FIX: was creating publisher inside method (wrong indentation + bad practice)
        self.cmd_vel_pub.publish(Twist())  # all zeros = stop
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------

    def make_goal(self, x, y, yaw):
        g = PoseStamped()
        g.header.frame_id = self.map_frame
        g.pose.position.x = x
        g.pose.position.y = y
        z, w = yaw_to_quat(yaw)
        g.pose.orientation.z = z
        g.pose.orientation.w = w
        return g

    def get_robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
            return pose_from_tf(tf, self.map_frame)
        except Exception:
            return None

    def euclidean(self, pose, goal):
        return math.hypot(
            pose.pose.position.x - goal.pose.position.x,
            pose.pose.position.y - goal.pose.position.y
        )

    # ------------------------------------------------------------------
    # Goal scoring — euclidean for far goals, path cost for nearby ones
    # ------------------------------------------------------------------

    def score_goal(self, robot, goal):
        dist = self.euclidean(robot, goal)
        if dist < self.OBSTACLE_REPLAN_RADIUS:
            # Close enough that obstacles matter — use real path cost
            cost = self.get_path_cost(robot, goal)
            return cost if cost is not None else float('inf')
        else:
            # Far away — straight line is fine, obstacles irrelevant
            return dist

    # ------------------------------------------------------------------
    # Path planning
    # ------------------------------------------------------------------

    def get_path_cost(self, start, goal):
        goal_msg = ComputePathToPose.Goal()
        goal_msg.start     = start
        goal_msg.goal      = goal
        goal_msg.use_start = True

        send_event   = threading.Event()
        result_event = threading.Event()
        gh_box  = [None]
        res_box = [None]

        def goal_cb(fut):
            gh_box[0] = fut.result()
            send_event.set()

        def result_cb(fut):
            res_box[0] = fut.result()
            result_event.set()

        sf = self.plan_client.send_goal_async(goal_msg)
        sf.add_done_callback(goal_cb)

        if not send_event.wait(timeout=5.0):
            return None
        gh = gh_box[0]
        if gh is None or not gh.accepted:
            return None

        rf = gh.get_result_async()
        rf.add_done_callback(result_cb)

        if not result_event.wait(timeout=5.0):
            return None
        res = res_box[0]
        if res is None:
            return None

        poses = res.result.path.poses
        if len(poses) < 2:
            return None

        dist = 0.0
        for i in range(1, len(poses)):
            x0, y0 = poses[i-1].pose.position.x, poses[i-1].pose.position.y
            x1, y1 = poses[i].pose.position.x,   poses[i].pose.position.y
            dist += math.hypot(x1 - x0, y1 - y0)
        return dist

    # ------------------------------------------------------------------
    # Costmap
    # ------------------------------------------------------------------

    def clear_costmaps(self):
        self.get_logger().info(">>> Clearing local costmap...")
        try:
            result = subprocess.run(
                ["ros2", "service", "call",
                 "/local_costmap/clear_entirely_local_costmap",
                 "nav2_msgs/srv/ClearEntireCostmap", "{}"],
                timeout=5,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.get_logger().info("    Local costmap cleared.")
            else:
                self.get_logger().warn("    Clear failed (continuing anyway).")
        except Exception:
            self.get_logger().warn("    Clear timed out (continuing anyway).")
        self.get_logger().info(">>> Waiting 3s to settle...")
        time.sleep(3.0)

    # ------------------------------------------------------------------
    # Send single navigation goal
    # ------------------------------------------------------------------

    def send_goal_once(self, goal):
        goal.header.stamp = self.get_clock().now().to_msg()
        ng = NavigateToPose.Goal()
        ng.pose = goal

        send_event   = threading.Event()
        result_event = threading.Event()
        gh_box  = [None]
        res_box = [None]

        def goal_cb(fut):
            gh_box[0] = fut.result()
            send_event.set()

        def result_cb(fut):
            res_box[0] = fut.result()
            result_event.set()

        sf = self.nav_client.send_goal_async(ng)
        sf.add_done_callback(goal_cb)

        if not send_event.wait(timeout=15.0):
            return 'timeout'
        gh = gh_box[0]
        if gh is None or not gh.accepted:
            return 'rejected'

        self.get_logger().info("Goal accepted! Driving...")
        rf = gh.get_result_async()
        rf.add_done_callback(result_cb)

        if not result_event.wait(timeout=180.0):
            return 'timeout'
        res = res_box[0]
        if res is not None and res.status == GoalStatus.STATUS_SUCCEEDED:
            return 'succeeded'
        return 'failed'

    # ------------------------------------------------------------------
    # Navigate with mid-drive replan check
    # ------------------------------------------------------------------

    def navigate_with_replan(self, goal):
        """
        Drive to goal. If it fails, check if the goal is still reachable.
        If unreachable → return False so navigation_loop can requeue + rescore.
        If reachable but just failed → clear costmaps and retry.
        """
        attempt = 0
        while True:
            attempt += 1
            self.get_logger().info(
                f"Sending goal ({goal.pose.position.x:.2f},{goal.pose.position.y:.2f}) attempt {attempt}..."
            )
            result = self.send_goal_once(goal)

            if result == 'succeeded':
                self.get_logger().info(f"Goal SUCCEEDED on attempt {attempt}!")
                return True

            elif result == 'failed':
                # Check if path to this goal still exists
                robot = self.get_robot_pose()
                if robot is not None:
                    cost = self.get_path_cost(robot, goal)
                    if cost is None:
                        self.get_logger().warn("Goal unreachable — triggering full rescore of all goals")
                        return False  # → navigation_loop will requeue + rescore
                self.get_logger().warn(f"Navigation FAILED (attempt {attempt}), path still exists. Clearing and retrying...")
                self.clear_costmaps()

            elif result == 'rejected':
                self.get_logger().warn(f"Goal REJECTED (attempt {attempt}). Clearing and retrying...")
                self.clear_costmaps()

            elif result == 'timeout':
                self.get_logger().warn(f"TIMEOUT (attempt {attempt}). Clearing and retrying...")
                self.clear_costmaps()

    # ------------------------------------------------------------------
    # Main navigation loop
    # ------------------------------------------------------------------

    def navigation_loop(self):
        self.get_logger().warn("Waiting for TF - set 2D Pose Estimate in RViz if needed!")
        while rclpy.ok():
            if self.get_robot_pose() is not None:
                break
            time.sleep(1.0)

        goal_num = 0
        while rclpy.ok() and self.goals:
            robot = self.get_robot_pose()
            if robot is None:
                self.get_logger().warn("Lost TF, waiting...")
                time.sleep(1.0)
                continue

            rx, ry = robot.pose.position.x, robot.pose.position.y
            goal_num += 1
            self.get_logger().info(f"===== GOAL {goal_num} | Robot at ({rx:.2f},{ry:.2f}) | {len(self.goals)} goals remaining =====")

            # Score all goals — euclidean for far, path cost for nearby
            scored = [(self.score_goal(robot, g), i) for i, g in enumerate(self.goals)]
            scored.sort()
            for score, i in scored:
                self.get_logger().info(
                    f"  #{i} ({self.goals[i].pose.position.x:.2f},{self.goals[i].pose.position.y:.2f}) score={score:.2f}m"
                )

            best_score, best_idx = scored[0]
            goal = self.goals.pop(best_idx)
            self.get_logger().info(
                f"GREEDY PICK: ({goal.pose.position.x:.2f},{goal.pose.position.y:.2f}) | {len(self.goals)} goals remaining after this"
            )

            self.clear_costmaps()

            # FIX: was calling both navigate_to() AND navigate_with_replan() — duplicate navigation removed
            # Now only navigate_with_replan() is used, which handles retries + replan check
            ok = self.navigate_with_replan(goal)

            if ok:
                robot_after = self.get_robot_pose()
                pos = f"({robot_after.pose.position.x:.2f},{robot_after.pose.position.y:.2f})" if robot_after else "unknown"
                self.get_logger().info(f"GOAL {goal_num} REACHED! Robot now at {pos}. {len(self.goals)} goals left.")
                self.get_logger().info("Waiting 5s for Nav2 to reset...")
                time.sleep(5.0)
                self.clear_costmaps()
            else:
                # Path was blocked mid-drive — requeue and rescore all goals next iteration
                self.get_logger().warn("Path blocked mid-drive! Requeueing goal and rescoring all goals...")
                self.goals.append(goal)
                # Loop continues → will rescore all goals including this one

        self.get_logger().info("ALL GOALS COMPLETED!")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    rclpy.init()
    node = Greedy4Goals()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info("Shutting down.")
        node.stop_robot()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()