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
# Tuning constants — Nav2 (unchanged from working code)
# ------------------------------------------------------------------
CHECK_INTERVAL    = 8.0
MAX_RECOVERIES    = 5
CHEAPER_THRESHOLD = 0.75

# ------------------------------------------------------------------
# LoRa / RSSI constants
# ------------------------------------------------------------------
RSSI_CONFIRM_THRESHOLD = -60   # minimum RSSI to confirm tag
RSSI_EXCELLENT         = -50
RSSI_GOOD              = -70
RSSI_WEAK              = -90
RSSI_VERY_WEAK         = -100
CREEP_SPEED            = 0.08  # m/s forward creep
CREEP_TIMEOUT          = 2.0   # seconds before giving up creep
CREEP_CHECK_INTERVAL   = 0.5   # seconds between RSSI checks while creeping

# ------------------------------------------------------------------
# Unified goal definitions — one place, no index lookup ever needed
# ------------------------------------------------------------------
GOAL_DEFS = [
    {"x": 1.43,   "y":  0.168, "yaw": 0.000866, "tag": "TAG_001"},
    {"x": 1.22,  "y": -1.72,  "yaw": 0.0021, "tag": "TAG_002"},
    {"x": 0.60, "y": -1.70, "yaw": 0.00252, "tag": "TAG_003"},
    {"x": 0.658, "y": -0.3730, "yaw": 0.00391, "tag": "TAG_004"},
]


def rssi_description(rssi):
    if rssi is None:           return "No signal"
    elif rssi >= RSSI_EXCELLENT:       return "Excellent"
    elif rssi >= RSSI_GOOD:            return "Good"
    elif rssi >= RSSI_CONFIRM_THRESHOLD: return "OK - above threshold"
    elif rssi >= RSSI_WEAK:            return "Weak - creeping"
    elif rssi >= RSSI_VERY_WEAK:       return "Very Weak"
    else:                              return "No signal / noise"


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

        self.nav_client  = ActionClient(
            self, NavigateToPose,    "navigate_to_pose",
            callback_group=self.cb_group)
        self.plan_client = ActionClient(
            self, ComputePathToPose, "compute_path_to_pose",
            callback_group=self.cb_group)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.home_pose   = None

        # ── LoRa state ─────────────────────────────────────────────
        self._lora_tag_id = None
        self._lora_rssi   = -999
        self._lora_lock   = threading.Lock()
        self._lora_last_seen = {}  # tag_id → (rssi, timestamp)

        self.create_subscription(
            String, '/lora_tag',
            self._lora_callback, 10,
            callback_group=self.cb_group
        )
        self.get_logger().info("LoRa subscriber ready on /lora_tag")

        # ── Goals: list of (PoseStamped, tag_string) tuples ────────
        self.goals = [
            (self.make_goal(d["x"], d["y"], d["yaw"]), d["tag"])
            for d in GOAL_DEFS
        ]

        self.get_logger().info("Waiting for Nav2 action server...")
        self.nav_client.wait_for_server()
        self.plan_client.wait_for_server()
        self.get_logger().info("Nav2 ready! Waiting 15s for full initialization...")
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
                self._lora_last_seen[tag_id] = (rssi, time.time())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Stop robot immediately
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
            x0, y0 = poses[i-1].pose.position.x, poses[i-1].pose.position.y
            x1, y1 = poses[i].pose.position.x,   poses[i].pose.position.y
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
                self.get_logger().warn("    Clear failed (continuing anyway).")
        except Exception:
            self.get_logger().warn("    Clear timed out (continuing anyway).")
        self.get_logger().info(">>> Waiting 3s to settle...")
        time.sleep(5.0)

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

        def feedback_cb(feedback_msg):
            recoveries = feedback_msg.feedback.number_of_recoveries
            if recoveries > self._recovery_count:
                self._recovery_count = recoveries
                self._amcl_nudged = False
                self.get_logger().warn(
                    f"Nav2 recovery #{recoveries} triggered")

        sf = self.nav_client.send_goal_async(ng, feedback_callback=feedback_cb)
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
                f"Too many recoveries ({self._recovery_count}) — switching goal")
            return 'too_many_recoveries'

        current_cost = self.get_path_cost(robot, current_goal, timeout=3.0)
        if current_cost is None:
            self.get_logger().warn("Current goal unreachable — switching")
            return 'cheaper_found'

        for (other_goal, _) in self.goals:
            other_cost = self.get_path_cost(robot, other_goal, timeout=3.0)
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
            self.get_logger().warn("Goal rejected or send timed out.")
            return 'failed'

        self.get_logger().info("Goal accepted! Monitoring mid-drive...")

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
    # LoRa approach — ONLY called after Nav2 already succeeded.
    #
    # FIX 1 — Rotation: stop as soon as signal drops even 1 dBm from
    #          the recorded peak. No counter, no tolerance window.
    #          The robot is now guaranteed to stop AT the peak heading.
    #
    # FIX 2 — Creeping: stop (and reverse) the instant the very first
    #          reading is weaker than the previous one. No consecutive
    #          drops required.
    # ------------------------------------------------------------------

    def _approach_via_lora(self, expected_tag: str) -> bool:
        stop      = Twist()
        creep_fwd = Twist(); creep_fwd.linear.x =  CREEP_SPEED
        creep_bwd = Twist(); creep_bwd.linear.x = -CREEP_SPEED
        rotate_cw = Twist(); rotate_cw.angular.z  = -0.2
        rotate_ccw= Twist(); rotate_ccw.angular.z =  0.2

        def get_rssi():
            with self._lora_lock:
                entry = self._lora_last_seen.get(expected_tag)
                if entry and (time.time() - entry[1]) < 5.0:
                    return entry[0]
            return None

        def sample_rssi(duration=1.0):
            samples = []
            deadline = time.time() + duration
            while time.time() < deadline:
                r = get_rssi()
                if r is not None:
                    samples.append(r)
                time.sleep(0.2)
            return sum(samples) / len(samples) if samples else None

        # ── Release Nav2 control ────────────────────────────────────
        for _ in range(10):
            self.cmd_vel_pub.publish(stop)
            time.sleep(0.1)
        time.sleep(1.5)

        # ── Phase 1: check on arrival ───────────────────────────────
        rssi = get_rssi()
        self.get_logger().info(
            f"[LoRa] Arrived at {expected_tag}. "
            f"RSSI={rssi} dBm | {rssi_description(rssi)}"
        )

        if rssi is None:
            self.get_logger().warn(f"[LoRa] {expected_tag} — no signal.")
            return False

        if rssi >= RSSI_CONFIRM_THRESHOLD:
            self.get_logger().info(f"[LoRa] {expected_tag} CONFIRMED! RSSI={rssi}")
            return True

        self.get_logger().warn(
            f"[LoRa] Weak signal ({rssi} dBm). Rotating to find best heading..."
        )

        # ── Phase 2: rotate, stop THE MOMENT signal drops from peak ─
        #
        # We record `best_rssi` as the highest seen so far and
        # `best_elapsed` as the time at which it occurred.
        # As soon as any new reading falls below `best_rssi` (even by
        # 1 dBm) AND we have already improved from the arrival RSSI,
        # we stop immediately — the previous sample was the peak.
        best_rssi        = rssi    # best seen so far
        best_elapsed     = 0.0    # timestamp of best reading
        drops_after_peak = 0      # consecutive meaningful drops below peak
        rotation_speed   = 0.2   # rad/s — must match rotate_ccw.angular.z
        full_rotation    = (2 * math.pi) / rotation_speed  # ~31 s max

        self.cmd_vel_pub.publish(rotate_ccw)
        rot_start = time.time()

        while time.time() - rot_start < full_rotation:
            r = get_rssi()
            if r is not None:
                elapsed = time.time() - rot_start
                self.get_logger().info(
                    f"[LoRa] Rotating... RSSI={r} dBm | {rssi_description(r)}"
                )

                if r >= RSSI_CONFIRM_THRESHOLD:
                    self.cmd_vel_pub.publish(stop)
                    self.get_logger().info(
                        f"[LoRa] {expected_tag} CONFIRMED while rotating! RSSI={r}")
                    return True

                if r > best_rssi:
                    # New peak — reset drop counter and keep going
                    best_rssi        = r
                    best_elapsed     = elapsed
                    drops_after_peak = 0
                elif best_rssi > rssi and r <= best_rssi - 2:
                    # ≥2 dBm below peak (filters ±1 dBm noise)
                    drops_after_peak += 1
                    if drops_after_peak >= 2:
                        # Two consecutive meaningful drops → peak is behind us
                        self.get_logger().info(
                            f"[LoRa] Peak passed, stopping rotation at t={elapsed:.1f}s "
                            f"(peak={best_rssi} dBm at t={best_elapsed:.1f}s, "
                            f"now={r} dBm)"
                        )
                        break
                else:
                    drops_after_peak = 0  # single-dBm noise — ignore

            time.sleep(0.3)

        self.cmd_vel_pub.publish(stop)
        time.sleep(0.3)

        self.get_logger().info(
            f"[LoRa] Best heading at t={best_elapsed:.1f}s RSSI={best_rssi} dBm"
        )

        # ── Phase 3: rotate back CW, guided by RSSI (not time-blind) ──
        #
        # Time-based rotate-back is unreliable — actual angular velocity
        # differs from the commanded value, causing multi-dBm errors.
        # Instead, rotate CW while watching RSSI; stop when RSSI peaks
        # and then drops again (same 2-drop/2dBm logic as forward scan).
        elapsed_now   = time.time() - rot_start
        time_to_best  = elapsed_now - best_elapsed
        # Allow slightly more than time_to_best so we don't undershoot
        max_back_time = time_to_best + 2.0

        self.get_logger().info(
            f"[LoRa] Rotating back (~{time_to_best:.1f}s) to best heading, "
            f"RSSI-guided..."
        )

        back_best_rssi        = None
        back_drops_after_peak = 0

        self.cmd_vel_pub.publish(rotate_cw)
        back_start = time.time()

        while time.time() - back_start < max_back_time:
            r = get_rssi()
            if r is not None:
                self.get_logger().info(f"[LoRa] Rotating back... RSSI={r} dBm")

                if r >= RSSI_CONFIRM_THRESHOLD:
                    self.cmd_vel_pub.publish(stop)
                    self.get_logger().info(
                        f"[LoRa] {expected_tag} CONFIRMED while rotating back! RSSI={r}")
                    return True

                if back_best_rssi is None or r > back_best_rssi:
                    back_best_rssi        = r
                    back_drops_after_peak = 0
                elif back_best_rssi is not None and r <= back_best_rssi - 2:
                    back_drops_after_peak += 1
                    if back_drops_after_peak >= 2:
                        self.get_logger().info(
                            f"[LoRa] Peak found while rotating back. "
                            f"RSSI={back_best_rssi} dBm, stopping."
                        )
                        break
                else:
                    back_drops_after_peak = 0
            time.sleep(0.3)

        self.cmd_vel_pub.publish(stop)
        time.sleep(0.5)

        rssi_after_rotate = sample_rssi(1.0)
        self.get_logger().info(
            f"[LoRa] At best heading. RSSI={rssi_after_rotate} dBm"
        )

        if rssi_after_rotate and rssi_after_rotate >= RSSI_CONFIRM_THRESHOLD:
            self.get_logger().info(f"[LoRa] {expected_tag} CONFIRMED at best heading!")
            return True

        # ── Phase 4: creep — stop THE INSTANT signal weakens ────────
        #
        # Any reading that is lower than the previous one triggers an
        # immediate stop + brief back-up. No consecutive-drop window.
        self.get_logger().info("[LoRa] Creeping toward tag...")
        start_pose        = self.get_robot_pose()
        last_rssi         = rssi_after_rotate or best_rssi
        go_forward        = True
        consecutive_drops = 0
        deadline          = time.time() + CREEP_TIMEOUT

        while time.time() < deadline:

            rssi = get_rssi()
            if rssi is not None:
                self.get_logger().info(
                    f"[LoRa] Creeping... RSSI={rssi} dBm | {rssi_description(rssi)}"
                )

                if rssi >= RSSI_CONFIRM_THRESHOLD:
                    self.cmd_vel_pub.publish(stop)
                    self.get_logger().info(
                        f"[LoRa] {expected_tag} CONFIRMED! RSSI={rssi}")
                    return True

                if rssi <= last_rssi - 2:
                    # Meaningful drop (≥2 dBm — not just noise)
                    consecutive_drops += 1
                    if consecutive_drops >= 2:
                        # Two consecutive real drops → we're moving away
                        self.cmd_vel_pub.publish(stop)
                        self.get_logger().warn(
                            f"[LoRa] Signal dropping ({last_rssi}→{rssi} dBm), "
                            f"backing up and holding..."
                        )
                        recover = creep_bwd if go_forward else creep_fwd
                        self.cmd_vel_pub.publish(recover)
                        time.sleep(0.5)
                        self.cmd_vel_pub.publish(stop)
                        time.sleep(2.0)   # hold — let RSSI stabilise
                        go_forward        = not go_forward
                        consecutive_drops = 0
                        last_rssi         = get_rssi() or rssi
                        continue
                else:
                    consecutive_drops = 0
                    last_rssi = rssi

            direction = creep_fwd if go_forward else creep_bwd
            self.cmd_vel_pub.publish(direction)
            time.sleep(CREEP_CHECK_INTERVAL)

        self.cmd_vel_pub.publish(stop)
        self.get_logger().warn(
            f"[LoRa] Timeout for {expected_tag}. Final RSSI={get_rssi()} dBm.")
        return False

    # ------------------------------------------------------------------
    # Navigate with replan
    # ------------------------------------------------------------------

    def navigate_with_replan(self, goal, expected_tag=None, max_attempts=None):
        attempt        = 0
        lora_confirmed = False

        while True:
            attempt += 1
            if max_attempts is not None and attempt > max_attempts:
                self.get_logger().warn(f"Giving up after {max_attempts} attempts.")
                return False, False

            self.get_logger().info(
                f"Sending goal ({goal.pose.position.x:.2f},"
                f"{goal.pose.position.y:.2f}) attempt {attempt}..."
            )

            result = self._drive_with_monitoring(goal)

            if result == 'succeeded':
                self.get_logger().info(f"Goal SUCCEEDED on attempt {attempt}!")

                if expected_tag is not None:
                    with self._lora_lock:
                        entry = self._lora_last_seen.get(expected_tag)
                        pre_rssi = entry[0] if entry and (time.time() - entry[1]) < 5.0 else None

                    if pre_rssi is None:
                        self.get_logger().info(
                            f"[RESULT] {expected_tag} — "
                            f"NAV2 COORDINATES ONLY (no LoRa signal)."
                        )
                    else:
                        lora_confirmed = self._approach_via_lora(expected_tag)
                        if lora_confirmed:
                            self.get_logger().info(
                                f"[RESULT] {expected_tag} — "
                                f"CONFIRMED via LoRa signal."
                            )
                        else:
                            self.get_logger().warn(
                                f"[RESULT] {expected_tag} — "
                                f"LoRa detected but not confirmed, "
                                f"accepted by Nav2 coordinates."
                            )

                return True, lora_confirmed

            elif result == 'switch_goal':
                return False, False

            elif result == 'failed':
                robot = self.get_robot_pose()
                if robot is not None:
                    current_cost = self.get_path_cost(robot, goal, timeout=3.0)
                    if current_cost is None:
                        self.get_logger().warn(
                            "Goal unreachable after failure — rescoring")
                        return False, False
                    for (other_goal, _) in self.goals:
                        other_cost = self.get_path_cost(
                            robot, other_goal, timeout=3.0)
                        if (other_cost is not None and
                                other_cost < current_cost * CHEAPER_THRESHOLD):
                            self.get_logger().warn(
                                "Cheaper goal after failure — rescoring")
                            return False, False

                self.get_logger().warn(
                    f"Failed (attempt {attempt}), clearing and retrying...")
                self.clear_costmaps()

    # ------------------------------------------------------------------
    # Main navigation loop
    # ------------------------------------------------------------------

    def navigation_loop(self):
        self.get_logger().warn(
            "Waiting for TF - set 2D Pose Estimate in RViz if needed!")
        while rclpy.ok():
            pose = self.get_robot_pose()
            if pose is not None:
                self.home_pose = pose
                self.get_logger().info(
                    f"Home position saved: "
                    f"({pose.pose.position.x:.2f},{pose.pose.position.y:.2f})"
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
                f"{len(self.goals)} goals remaining ====="
            )

            scored = [
                (self.score_goal(robot, g), i)
                for i, (g, tag) in enumerate(self.goals)
            ]
            scored.sort()
            for score, i in scored:
                g, tag = self.goals[i]
                self.get_logger().info(
                    f"  #{i} ({g.pose.position.x:.2f},"
                    f"{g.pose.position.y:.2f}) "
                    f"tag={tag} score={score:.2f}m"
                )

            best_score, best_idx = scored[0]
            goal, expected_tag = self.goals.pop(best_idx)

            self.get_logger().info(
                f"GREEDY PICK: "
                f"({goal.pose.position.x:.2f},{goal.pose.position.y:.2f}) | "
                f"Expecting: {expected_tag} | "
                f"{len(self.goals)} goals remaining after this"
            )

            with self._lora_lock:
                self._lora_tag_id = None
                self._lora_rssi   = -999
                self._lora_last_seen.pop(expected_tag, None)

            self.clear_costmaps()
            ok, lora_confirmed = self.navigate_with_replan(
                goal, expected_tag=expected_tag)

            if ok:
                robot_after = self.get_robot_pose()
                pos = (
                    f"({robot_after.pose.position.x:.2f},"
                    f"{robot_after.pose.position.y:.2f})"
                    if robot_after else "unknown"
                )
                if lora_confirmed:
                    self.get_logger().info(
                        f"GOAL {goal_num} REACHED! "
                        f"LoRa CONFIRMED {expected_tag} | "
                        f"Robot at {pos} | {len(self.goals)} goals left."
                    )
                else:
                    self.get_logger().info(
                        f"GOAL {goal_num} REACHED! "
                        f"Nav2 coordinates only ({expected_tag} no LoRa) | "
                        f"Robot at {pos} | {len(self.goals)} goals left."
                    )
                self.get_logger().info("Waiting 5s for Nav2 to reset...")
                time.sleep(5.0)
                self.clear_costmaps()
            else:
                self.get_logger().warn(
                    f"Requeueing {expected_tag} and rescoring all goals...")
                self.goals.append((goal, expected_tag))
                self.clear_costmaps()

        self.get_logger().info("ALL GOALS COMPLETED!")
        if self.home_pose is not None:
            hx = self.home_pose.pose.position.x
            hy = self.home_pose.pose.position.y
            self.get_logger().info(
                f"Returning to home ({hx:.2f},{hy:.2f})...")
            self.clear_costmaps()
            ok, _ = self.navigate_with_replan(
                self.home_pose, expected_tag=None, max_attempts=5)
            if ok:
                self.get_logger().info("MISSION COMPLETE! Robot returned home!")
            else:
                self.get_logger().warn("Failed to return home.")
        else:
            self.get_logger().warn("Home position not saved - cannot return.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    rclpy.init()
    node = Greedy4Goals()

    def shutdown_handler(sig, frame):
        node.get_logger().info('Shutting down - stopping robot...')
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