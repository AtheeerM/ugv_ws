#!/usr/bin/env python3
import math, time, threading, subprocess, rclpy, rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose, ComputePathToPose
from action_msgs.msg import GoalStatus
from std_msgs.msg import String
import tf2_ros
import signal
import sys
from geometry_msgs.msg import PoseWithCovarianceStamped

# ------------------------------------------------------------------
# Tuning constants — Nav2
# ------------------------------------------------------------------
CHECK_INTERVAL    = 8.0
MAX_RECOVERIES    = 5
CHEAPER_THRESHOLD = 1.0

# ------------------------------------------------------------------
# RSSI Signal Strength Thresholds
# -30 dBm = Excellent (modules touching / very close)
# -50 dBm = Very Good (close range, clear line of sight)
# -70 dBm = Good (normal indoor range)
# -80 dBm = OK  ← confirmation threshold
# -90 dBm = Weak (robot needs to creep closer)
# -100 dBm = Very Weak (barely detectable)
# -120 dBm = Terrible (almost out of range)
# -164 dBm = No signal (noise floor / not your tag)
# ------------------------------------------------------------------
RSSI_CONFIRM_THRESHOLD = -80
RSSI_EXCELLENT         = -50
RSSI_GOOD              = -70
RSSI_WEAK              = -90
RSSI_VERY_WEAK         = -100
CREEP_SPEED            = 0.08
CREEP_TIMEOUT          = 15.0
CREEP_CHECK_INTERVAL   = 0.5
MAX_CREEP_DISTANCE     = 1.5

# ------------------------------------------------------------------
# Tag-to-goal mapping
# TAG number = physical location (NOT visit order)
# Robot visits in greedy order but confirms with correct tag
# ------------------------------------------------------------------
GOAL_TAG_MAP = {
    0: "TAG_001",   # Location A: (-2.50, 7.00)
    1: "TAG_002",   # Location B: (-3.58, 0.56)
    2: "TAG_003",   # Location C: ( 3.73, 6.49)
    3: "TAG_004",   # Location D: (-2.50, 3.50)
}


def rssi_description(rssi):
    """Return human-readable RSSI signal description."""
    if rssi is None:
        return "No signal"
    elif rssi >= RSSI_EXCELLENT:
        return "Excellent"
    elif rssi >= RSSI_GOOD:
        return "Good"
    elif rssi >= RSSI_CONFIRM_THRESHOLD:
        return "OK - above threshold"
    elif rssi >= RSSI_WEAK:
        return "Weak - below threshold, creeping"
    elif rssi >= RSSI_VERY_WEAK:
        return "Very Weak"
    else:
        return "No signal / noise"


def yaw_to_quat(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def pose_from_tf(tf_stamped, frame_id):
    t = tf_stamped.transform
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.header.stamp    = tf_stamped.header.stamp
    ps.pose.position.x = t.translation.x
    ps.pose.position.y = t.translation.y
    ps.pose.position.z = t.translation.z
    ps.pose.orientation.x = t.rotation.x
    ps.pose.orientation.y = t.rotation.y
    ps.pose.orientation.z = t.rotation.z
    ps.pose.orientation.w = t.rotation.w
    return ps


class Greedy4Goals(Node):

    OBSTACLE_REPLAN_RADIUS = 2.0

    def __init__(self):
        super().__init__("greedy_4_goals")
        self.map_frame  = "map"
        self.base_frame = "base_footprint"
        self.cb_group        = ReentrantCallbackGroup()
        self._shutdown       = False
        self._current_gh     = None
        self._recovery_count = 0
        self._amcl_nudged    = False

        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        self.nav_client = ActionClient(
            self, NavigateToPose, "navigate_to_pose",
            callback_group=self.cb_group)
        self.plan_client = ActionClient(
            self, ComputePathToPose, "compute_path_to_pose",
            callback_group=self.cb_group)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.home_pose   = None

        # ── LoRa state ────────────────────────────────────────────
        self._lora_tag_id = None
        self._lora_rssi   = -999
        self._lora_lock   = threading.Lock()

        self.create_subscription(
            String, '/lora_tag',
            self._lora_callback, 10,
            callback_group=self.cb_group
        )
        self.get_logger().info("LoRa subscriber ready on /lora_tag")

        # ── Goals — TAG permanently tied to physical location ─────
        self.goals = [
            self.make_goal(2.89,   0.0335,   0.00442),
            self.make_goal(2.92,   3.95,    0.00013),
            self.make_goal(1.1,  3.01,     0.00149),
            self.make_goal(0.998,  2.0,  0.0056),
        ]

        self.get_logger().info("Waiting for Nav2 action server...")
        self.nav_client.wait_for_server()
        self.plan_client.wait_for_server()
        self.get_logger().info("Nav2 ready! Waiting 15s...")
        time.sleep(15.0)
        self.get_logger().info("Starting greedy navigation!")

        self._nav_thread = threading.Thread(
            target=self.navigation_loop, daemon=True)
        self._nav_thread.start()

    # ------------------------------------------------------------------
    # LoRa callback — expects "TAG_001,-73" from lora_serial_node.py
    # ------------------------------------------------------------------

    def _lora_callback(self, msg: String):
        try:
            parts  = msg.data.split(',')
            tag_id = parts[0].strip()
            rssi   = int(parts[1].strip()) if len(parts) > 1 else -999
            with self._lora_lock:
                self._lora_tag_id = tag_id
                self._lora_rssi   = rssi
            self.get_logger().info(
                f"LoRa received: {tag_id} | "
                f"RSSI: {rssi} dBm | "
                f"Signal: {rssi_description(rssi)}"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Stop robot
    # ------------------------------------------------------------------

    def stop_robot(self):
        self._shutdown = True
        if self._current_gh is not None:
            try:
                cancel_future = self._current_gh.cancel_goal_async()
                deadline = time.time() + 1.0
                while not cancel_future.done() and time.time() < deadline:
                    time.sleep(0.05)
            except Exception:
                pass
            self._current_gh = None
        for _ in range(5):
            self.cmd_vel_pub.publish(Twist())
            time.sleep(0.05)

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
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            return pose_from_tf(tf, self.map_frame)
        except Exception:
            return None

    def euclidean(self, pose, goal):
        return math.hypot(
            pose.pose.position.x - goal.pose.position.x,
            pose.pose.position.y - goal.pose.position.y
        )

    # ------------------------------------------------------------------
    # Goal scoring
    # ------------------------------------------------------------------

    def score_goal(self, robot, goal):
        dist = self.euclidean(robot, goal)
        if dist < self.OBSTACLE_REPLAN_RADIUS:
            cost = self.get_path_cost(robot, goal)
            return cost if cost is not None else float('inf')
        return dist

    # ------------------------------------------------------------------
    # Path planning
    # ------------------------------------------------------------------

    def get_path_cost(self, start, goal, timeout=5.0):
        goal_msg           = ComputePathToPose.Goal()
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

        if not send_event.wait(timeout=timeout):
            return None
        gh = gh_box[0]
        if gh is None or not gh.accepted:
            return None

        rf = gh.get_result_async()
        rf.add_done_callback(result_cb)

        if not result_event.wait(timeout=timeout):
            return None
        res = res_box[0]
        if res is None:
            return None

        poses = res.result.path.poses
        if len(poses) < 2:
            return None

        dist = 0.0
        for i in range(1, len(poses)):
            x0 = poses[i-1].pose.position.x
            y0 = poses[i-1].pose.position.y
            x1 = poses[i].pose.position.x
            y1 = poses[i].pose.position.y
            dist += math.hypot(x1 - x0, y1 - y0)
        return dist

    # ------------------------------------------------------------------
    # AMCL relocalization nudge
    # ------------------------------------------------------------------

    def trigger_amcl_relocalize(self):
        pose = self.get_robot_pose()
        if pose is None:
            return
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id    = "map"
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.pose.pose          = pose.pose
        msg.pose.covariance[0]  = 0.25
        msg.pose.covariance[7]  = 0.25
        msg.pose.covariance[35] = 0.1
        self.initial_pose_pub.publish(msg)
        self.get_logger().info("AMCL relocalization triggered")

    # ------------------------------------------------------------------
    # Costmap clear
    # ------------------------------------------------------------------

    def clear_costmaps(self):
        self.get_logger().info(">>> Clearing local costmap...")
        try:
            result = subprocess.run(
                ["ros2", "service", "call",
                 "/local_costmap/clear_entirely_local_costmap",
                 "nav2_msgs/srv/ClearEntireCostmap", "{}"],
                timeout=5, capture_output=True, text=True
            )
            if result.returncode == 0:
                self.get_logger().info("    Local costmap cleared.")
            else:
                self.get_logger().warn("    Clear failed.")
        except Exception:
            self.get_logger().warn("    Clear timed out.")
        self.get_logger().info(">>> Waiting 3s to settle...")
        time.sleep(3.0)

    # ------------------------------------------------------------------
    # Cancel active Nav2 goal
    # ------------------------------------------------------------------

    def _cancel_current_goal(self, reason=""):
        if self._current_gh is None:
            return
        self.get_logger().warn(f"Cancelling current goal: {reason}")
        try:
            cancel_future = self._current_gh.cancel_goal_async()
            deadline = time.time() + 2.0
            while not cancel_future.done() and time.time() < deadline:
                time.sleep(0.05)
        except Exception:
            pass
        self._current_gh = None

    # ------------------------------------------------------------------
    # Send goal asynchronously
    # ------------------------------------------------------------------

    def _send_goal_async(self, goal):
        goal.header.stamp = self.get_clock().now().to_msg()
        ng      = NavigateToPose.Goal()
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

        def feedback_cb(feedback_msg):
            recoveries = feedback_msg.feedback.number_of_recoveries
            if recoveries > self._recovery_count:
                self._recovery_count = recoveries
                self._amcl_nudged    = False
                self.get_logger().warn(
                    f"Nav2 recovery #{recoveries} triggered")

        sf = self.nav_client.send_goal_async(
            ng, feedback_callback=feedback_cb)
        sf.add_done_callback(goal_cb)

        if not send_event.wait(timeout=15.0):
            return None, None, None

        gh = gh_box[0]
        if gh is None or not gh.accepted:
            return None, None, None

        self._current_gh     = gh
        self._recovery_count = 0
        self._amcl_nudged    = False

        rf = gh.get_result_async()
        rf.add_done_callback(result_cb)

        return gh, result_event, res_box

    # ------------------------------------------------------------------
    # Mid-drive check
    # ------------------------------------------------------------------

    def _mid_drive_check(self, current_goal):
        robot = self.get_robot_pose()
        if robot is None:
            return 'continue'

        if self._recovery_count >= 1 and not self._amcl_nudged:
            self.trigger_amcl_relocalize()
            self._amcl_nudged = True

        if self._recovery_count >= MAX_RECOVERIES:
            self.get_logger().warn(
                f"Too many recoveries ({self._recovery_count})")
            return 'too_many_recoveries'

        current_cost = self.get_path_cost(robot, current_goal, timeout=3.0)
        if current_cost is None:
            self.get_logger().warn("Current goal unreachable")
            return 'cheaper_found'

        for other in self.goals:
            other_cost = self.get_path_cost(robot, other, timeout=3.0)
            if other_cost is not None and \
                    other_cost < current_cost * CHEAPER_THRESHOLD:
                return 'cheaper_found'

        return 'continue'

    # ------------------------------------------------------------------
    # Drive with monitoring
    # ------------------------------------------------------------------

    def _drive_with_monitoring(self, goal):
        gh, result_event, res_box = self._send_goal_async(goal)

        if gh is None:
            self.get_logger().warn("Goal rejected or timed out.")
            return 'failed'

        self.get_logger().info("Goal accepted! Monitoring...")

        while not result_event.wait(timeout=CHECK_INTERVAL):
            decision = self._mid_drive_check(goal)
            if decision != 'continue':
                self._cancel_current_goal(reason=decision)
                return 'switch_goal'

        res = res_box[0]
        if res is not None and res.status == GoalStatus.STATUS_SUCCEEDED:
            return 'succeeded'
        return 'failed'

    # ------------------------------------------------------------------
    # LoRa-guided approach (called after Nav2 succeeds)
    # ------------------------------------------------------------------

    def _approach_via_lora(self, goal_idx: int) -> bool:
        """
        After Nav2 reaches coordinate, check LoRa tag.

        Three cases:
          rssi = None          → no signal at all → return False
                                  (caller prints 'reached without LoRa')
          rssi < threshold     → signal detected but weak → creep closer
          rssi >= threshold    → confirmed immediately → return True

        RSSI reference:
          >= -50   Excellent
          >= -70   Good
          >= -80   OK  ← confirmation threshold
          >= -90   Weak  (creep)
          >= -100  Very Weak  (creep, may not help)
          <  -100  No signal
        """
        expected_tag = GOAL_TAG_MAP.get(goal_idx)
        if expected_tag is None:
            self.get_logger().warn(
                f"No tag for goal {goal_idx}, skipping LoRa.")
            return True

        stop  = Twist()
        creep = Twist()
        creep.linear.x = CREEP_SPEED

        def get_rssi():
            with self._lora_lock:
                if self._lora_tag_id == expected_tag:
                    return self._lora_rssi
            return None

        # ── Phase 1: check immediately after arriving ─────────────
        rssi = get_rssi()
        desc = rssi_description(rssi)

        self.get_logger().info(
            f"[LoRa] Coordinate reached. "
            f"Checking {expected_tag}... "
            f"RSSI={rssi} dBm | Signal: {desc}"
        )

        # Case A: no signal at all — do not creep, let caller handle it
        if rssi is None:
            self.get_logger().warn(
                f"[LoRa] {expected_tag} — no signal detected.")
            return False

        # Case B: signal strong enough — confirm immediately
        if rssi >= RSSI_CONFIRM_THRESHOLD:
            self.get_logger().info(
                f"[LoRa] {expected_tag} CONFIRMED! "
                f"RSSI={rssi} dBm ({desc})"
            )
            return True

        # Case C: signal detected but too weak — creep to get closer
        self.get_logger().warn(
            f"[LoRa] {expected_tag} detected but weak: "
            f"RSSI={rssi} dBm ({desc}). "
            f"Need >= {RSSI_CONFIRM_THRESHOLD} dBm. "
            f"Creeping forward..."
        )

        # ── Phase 2: creep toward tag ─────────────────────────────
        start_pose = self.get_robot_pose()
        last_rssi  = rssi
        deadline   = time.time() + CREEP_TIMEOUT

        while time.time() < deadline:

            # Distance guard
            current_pose = self.get_robot_pose()
            if start_pose and current_pose:
                crept = self.euclidean(start_pose, current_pose)
                if crept > MAX_CREEP_DISTANCE:
                    self.cmd_vel_pub.publish(stop)
                    self.get_logger().warn(
                        f"[LoRa] Crept {crept:.2f}m — max distance reached.")
                    return False

            rssi = get_rssi()

            if rssi is not None:
                desc = rssi_description(rssi)
                self.get_logger().info(
                    f"[LoRa] Creeping... "
                    f"{expected_tag} RSSI={rssi} dBm | {desc}"
                )

                # Confirmed while creeping
                if rssi >= RSSI_CONFIRM_THRESHOLD:
                    self.cmd_vel_pub.publish(stop)
                    self.get_logger().info(
                        f"[LoRa] {expected_tag} CONFIRMED while creeping! "
                        f"RSSI={rssi} dBm ({desc})"
                    )
                    return True

                # RSSI dropped — rotate briefly to reacquire
                if rssi < last_rssi - 5:
                    self.cmd_vel_pub.publish(stop)
                    self.get_logger().warn(
                        f"[LoRa] RSSI dropped "
                        f"({last_rssi}→{rssi} dBm) — rotating...")
                    rotate = Twist()
                    rotate.angular.z = 0.3
                    for _ in range(6):      # ~1 second
                        self.cmd_vel_pub.publish(rotate)
                        time.sleep(0.2)

                last_rssi = rssi

            # Keep creeping
            self.cmd_vel_pub.publish(creep)
            time.sleep(CREEP_CHECK_INTERVAL)

        # ── Timeout ───────────────────────────────────────────────
        self.cmd_vel_pub.publish(stop)
        final_rssi = get_rssi()
        self.get_logger().warn(
            f"[LoRa] Creep timeout for {expected_tag}. "
            f"Final RSSI={final_rssi} dBm "
            f"({rssi_description(final_rssi)})."
        )
        return False

    # ------------------------------------------------------------------
    # Navigate with replan
    # ------------------------------------------------------------------

    def navigate_with_replan(self, goal, goal_idx=None, max_attempts=None):
        attempt = 0
        lora_confirmed = False  # ← track this

        while True:
            attempt += 1
            if max_attempts is not None and attempt > max_attempts:
                self.get_logger().warn(f"Giving up after {max_attempts} attempts.")
                return False, False  # ← return tuple

            x = goal.pose.position.x
            y = goal.pose.position.y
            self.get_logger().info(f"Sending goal ({x:.2f},{y:.2f}) attempt {attempt}...")

            result = self._drive_with_monitoring(goal)

            if result == 'succeeded':
                self.get_logger().info(f"Nav2 SUCCEEDED on attempt {attempt}!")

                if goal_idx is not None:
                    with self._lora_lock:
                        pre_rssi = (
                            self._lora_rssi
                            if self._lora_tag_id == GOAL_TAG_MAP.get(goal_idx)
                            else None
                        )

                    if pre_rssi is None:
                        # ── OUTCOME 1: Nav2 only ──────────────────────
                        self.get_logger().info(
                            f"[RESULT] Destination {goal_idx} ({GOAL_TAG_MAP.get(goal_idx)}) "
                            f"reached by NAV2 COORDINATES ONLY — no LoRa signal detected."
                        )
                    else:
                        lora_confirmed = self._approach_via_lora(goal_idx)
                        if lora_confirmed:
                            # ── OUTCOME 2: LoRa guided ────────────────
                            self.get_logger().info(
                                f"[RESULT] Destination {goal_idx} ({GOAL_TAG_MAP.get(goal_idx)}) "
                                f"reached and CONFIRMED via LoRa — "
                                f"robot closed in using signal strength."
                            )
                        else:
                            # ── Signal detected but couldn't confirm ──
                            self.get_logger().warn(
                                f"[RESULT] Destination {goal_idx} ({GOAL_TAG_MAP.get(goal_idx)}) "
                                f"LoRa signal was detected but could not reach threshold — "
                                f"marked done by Nav2 coordinates."
                            )

                return True, lora_confirmed  # ← return tuple

            elif result == 'switch_goal':
                return False, False

            elif result == 'failed':
                robot = self.get_robot_pose()
                if robot is not None:
                    current_cost = self.get_path_cost(robot, goal, timeout=3.0)
                    if current_cost is None:
                        return False, False
                    for other in self.goals:
                        other_cost = self.get_path_cost(robot, other, timeout=3.0)
                        if (other_cost is not None and
                                other_cost < current_cost * CHEAPER_THRESHOLD):
                            return False, False

                self.get_logger().warn(f"Failed attempt {attempt}, retrying...")
                self.clear_costmaps()

    # ------------------------------------------------------------------
    # Main navigation loop
    # ------------------------------------------------------------------

    def navigation_loop(self):
        self.get_logger().warn(
            "Waiting for TF — set 2D Pose Estimate in RViz!")
        while rclpy.ok():
            pose = self.get_robot_pose()
            if pose is not None:
                self.home_pose = pose
                self.get_logger().info(
                    f"Home saved: "
                    f"({pose.pose.position.x:.2f},"
                    f"{pose.pose.position.y:.2f})"
                )
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
            self.get_logger().info(
                f"===== GOAL {goal_num} | "
                f"Robot at ({rx:.2f},{ry:.2f}) | "
                f"{len(self.goals)} remaining ====="
            )

            scored = [
                (self.score_goal(robot, g), i)
                for i, g in enumerate(self.goals)
            ]
            scored.sort()
            for score, i in scored:
                self.get_logger().info(
                    f"  #{i} ({self.goals[i].pose.position.x:.2f},"
                    f"{self.goals[i].pose.position.y:.2f}) "
                    f"score={score:.2f}m"
                )

            best_score, best_idx = scored[0]
            goal = self.goals.pop(best_idx)

            # Resolve original index for TAG mapping
            goal_coords = [
                (2.92, 3.95),   # TAG_001
                (1.1, 3.01),   # TAG_002
                ( 0.998, 2.0),   # TAG_003
                (2.89, 0.0335),   # TAG_004
            ]
            original_idx = None
            for orig_i, (x, y) in enumerate(goal_coords):
                if (abs(goal.pose.position.x - x) < 0.01 and
                        abs(goal.pose.position.y - y) < 0.01):
                    original_idx = orig_i
                    break

            expected_tag = GOAL_TAG_MAP.get(original_idx, "Unknown")
            self.get_logger().info(
                f"GREEDY PICK: "
                f"({goal.pose.position.x:.2f},"
                f"{goal.pose.position.y:.2f}) | "
                f"Expecting: {expected_tag} | "
                f"{len(self.goals)} remaining"
            )
            # Reset LoRa state before approaching new goal
            with self._lora_lock:
                self._lora_tag_id = None
                self._lora_rssi   = -999

            self.clear_costmaps()
            ok, lora_confirmed = self.navigate_with_replan(goal, goal_idx=original_idx)
            if ok:
                robot_after = self.get_robot_pose()
                pos = (
                    f"({robot_after.pose.position.x:.2f},"
                    f"{robot_after.pose.position.y:.2f})"
                    if robot_after else "unknown"
                )
                if lora_confirmed:
                    self.get_logger().info(
                        f"===== GOAL {goal_num} COMPLETE | "
                        f"LoRa guided approach — robot closed in on {expected_tag} | "
                        f"Robot at {pos} | {len(self.goals)} left ====="
                    )
                else:
                    self.get_logger().info(
                        f"===== GOAL {goal_num} COMPLETE | "
                        f"Nav2 coordinates only — no LoRa signal from {expected_tag} | "
                        f"Robot at {pos} | {len(self.goals)} left ====="
                    )
                self.get_logger().info("Waiting 5s to reset...")
                time.sleep(5.0)
                self.clear_costmaps()
            else:
                self.get_logger().warn("Requeueing goal...")
                self.goals.append(goal)
                self.clear_costmaps()

        self.get_logger().info("ALL GOALS COMPLETED!")
        if self.home_pose is not None:
            hx = self.home_pose.pose.position.x
            hy = self.home_pose.pose.position.y
            self.get_logger().info(
                f"Returning home ({hx:.2f},{hy:.2f})...")
            self.clear_costmaps()
            ok, _ = self.navigate_with_replan(            # ← unpack tuple, ignore lora for home
                self.home_pose, goal_idx=None, max_attempts=5)
            if ok:
                self.get_logger().info(
                    "MISSION COMPLETE! Robot returned home!")
            else:
                self.get_logger().warn("Failed to return home.")
        else:
            self.get_logger().warn("Home not saved — cannot return.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    rclpy.init()
    node = Greedy4Goals()

    def shutdown_handler(sig, frame):
        node.get_logger().info('Shutting down...')
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()


if __name__ == "__main__":
    main()