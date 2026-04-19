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
CREEP_CHECK_INTERVAL   = 0.5   # seconds per forward step while creeping

# ------------------------------------------------------------------
# Unified goal definitions — one place, no index lookup ever needed
# ------------------------------------------------------------------
GOAL_DEFS = [
    {"x": 3.1,   "y":  0.513, "yaw": 0.00335, "tag": "TAG_001"},
    {"x": 1.88,  "y": -1.91,  "yaw": 0.00206, "tag": "TAG_002"},
    {"x": 3.72, "y": -4.22, "yaw": 0.00218, "tag": "TAG_003"},
    {"x": 0.899, "y": -4.03, "yaw": 0.00666, "tag": "TAG_004"},
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

        # ── Phase 2: quick coarse probe to pick the better side ──────
        #
        # Probe CCW 3s (~34°) → sample.
        # Probe CW 6s (back + 3s past) → sample.
        # Pick the side with stronger signal.
        # Then fine-scan only that side until peak passes.
        COARSE_TIME    = 3.0   # seconds per coarse probe (~34° at 0.2 rad/s)
        FINE_TIME      = 8.0   # max seconds for fine scan (~90°)
        rotation_speed = 0.2

        # — Probe CCW —
        self.cmd_vel_pub.publish(rotate_ccw)
        time.sleep(COARSE_TIME)
        self.cmd_vel_pub.publish(stop)
        time.sleep(0.3)
        rssi_ccw = sample_rssi(0.5)
        self.get_logger().info(f"[LoRa] Coarse CCW: RSSI={rssi_ccw} dBm")

        # — Probe CW (swing back past start by COARSE_TIME) —
        self.cmd_vel_pub.publish(rotate_cw)
        time.sleep(COARSE_TIME * 2)
        self.cmd_vel_pub.publish(stop)
        time.sleep(0.3)
        rssi_cw = sample_rssi(0.5)
        self.get_logger().info(f"[LoRa] Coarse CW:  RSSI={rssi_cw} dBm")

        # — Return to start heading —
        self.cmd_vel_pub.publish(rotate_ccw)
        time.sleep(COARSE_TIME)
        self.cmd_vel_pub.publish(stop)
        time.sleep(0.3)

        # — Pick the stronger side —
        if rssi_ccw is not None and (rssi_cw is None or rssi_ccw >= rssi_cw):
            fine_dir   = rotate_ccw
            fine_label = "CCW"
            best_rssi  = rssi_ccw
        else:
            fine_dir   = rotate_cw
            fine_label = "CW"
            best_rssi  = rssi_cw

        best_rssi = max(r for r in [rssi, best_rssi] if r is not None)
        self.get_logger().info(
            f"[LoRa] Chosen side: {fine_label} "
            f"(CCW={rssi_ccw} dBm, CW={rssi_cw} dBm)"
        )

        # ── Phase 3: fine scan on chosen side only ────────────────────
        #
        # Rotate in the chosen direction, stop when peak passes
        # (2 consecutive drops of ≥2 dBm). Max FINE_TIME seconds.
        drops_after_peak = 0
        self.cmd_vel_pub.publish(fine_dir)
        fine_start = time.time()

        while time.time() - fine_start < FINE_TIME:
            r = get_rssi()
            if r is not None:
                elapsed = time.time() - fine_start
                self.get_logger().info(
                    f"[LoRa] Fine {fine_label}... RSSI={r} dBm | {rssi_description(r)}"
                )

                if r >= RSSI_CONFIRM_THRESHOLD:
                    self.cmd_vel_pub.publish(stop)
                    self.get_logger().info(
                        f"[LoRa] {expected_tag} CONFIRMED during fine scan! RSSI={r}")
                    return True

                if r > best_rssi:
                    best_rssi        = r
                    drops_after_peak = 0
                elif best_rssi > rssi and r <= best_rssi - 2:
                    drops_after_peak += 1
                    if drops_after_peak >= 2:
                        self.get_logger().info(
                            f"[LoRa] Peak found at t={elapsed:.1f}s, "
                            f"peak={best_rssi} dBm, now={r} dBm"
                        )
                        break
                else:
                    drops_after_peak = 0

            time.sleep(0.3)

        self.cmd_vel_pub.publish(stop)
        time.sleep(0.3)
        self.get_logger().info(
            f"[LoRa] Best heading found. Peak RSSI={best_rssi} dBm"
        )

        rssi_at_heading = sample_rssi(1.0)
        self.get_logger().info(
            f"[LoRa] At best heading. RSSI={rssi_at_heading} dBm"
        )

        if rssi_at_heading and rssi_at_heading >= RSSI_CONFIRM_THRESHOLD:
            self.get_logger().info(f"[LoRa] {expected_tag} CONFIRMED at best heading!")
            return True

        # ── Phase 4: align → burst → re-align if signal drops ───────
        #
        # Each cycle:
        #   1. Probe ±ALIGN_TIME to find the strongest heading
        #   2. Drive forward in short steps, checking RSSI each step
        #   3. Confirm immediately if RSSI ≥ threshold
        #   4. If signal drops → back up, re-align, try again
        #   5. Never drive forward when signal is getting weaker
        ALIGN_TIME      = 3.0    # seconds per probe direction (~34° at 0.2 rad/s)
        STEP_TIME       = 0.5    # seconds per forward step
        BURST_MAX_STEPS = 5      # max steps per burst before re-aligning
        MAX_CYCLES      = 5      # max align+burst attempts before giving up

        current_rssi = rssi_at_heading or best_rssi

        for cycle in range(MAX_CYCLES):
            self.get_logger().info(
                f"[LoRa] Cycle {cycle+1}/{MAX_CYCLES} — aligning heading..."
            )

            # ── 1. Probe ± to find best heading ──────────────────────
            self.cmd_vel_pub.publish(rotate_ccw)
            time.sleep(ALIGN_TIME)
            self.cmd_vel_pub.publish(stop)
            time.sleep(0.2)
            probe_ccw = sample_rssi(0.3)

            self.cmd_vel_pub.publish(rotate_cw)
            time.sleep(ALIGN_TIME * 2)
            self.cmd_vel_pub.publish(stop)
            time.sleep(0.2)
            probe_cw = sample_rssi(0.3)

            # Return to centre
            self.cmd_vel_pub.publish(rotate_ccw)
            time.sleep(ALIGN_TIME)
            self.cmd_vel_pub.publish(stop)
            time.sleep(0.2)

            self.get_logger().info(
                f"[LoRa] Probe — CCW={probe_ccw} dBm, CW={probe_cw} dBm, "
                f"centre={current_rssi} dBm"
            )

            # Check confirm during probing
            for p in [probe_ccw, probe_cw]:
                if p is not None and p >= RSSI_CONFIRM_THRESHOLD:
                    self.get_logger().info(
                        f"[LoRa] {expected_tag} CONFIRMED during alignment! RSSI={p}")
                    return True

            # Rotate toward the strongest direction if it beats centre
            best_probe = max(
                (r for r in [probe_ccw, probe_cw] if r is not None),
                default=current_rssi
            )
            if best_probe > current_rssi:
                if probe_ccw is not None and probe_ccw == best_probe:
                    self.cmd_vel_pub.publish(rotate_ccw)
                    time.sleep(ALIGN_TIME)
                    self.cmd_vel_pub.publish(stop)
                    self.get_logger().info(f"[LoRa] Turned CCW → RSSI={probe_ccw} dBm")
                    current_rssi = probe_ccw
                elif probe_cw is not None and probe_cw == best_probe:
                    self.cmd_vel_pub.publish(rotate_cw)
                    time.sleep(ALIGN_TIME)
                    self.cmd_vel_pub.publish(stop)
                    self.get_logger().info(f"[LoRa] Turned CW → RSSI={probe_cw} dBm")
                    current_rssi = probe_cw
            else:
                self.get_logger().info("[LoRa] Heading optimal, no rotation needed.")

            time.sleep(0.2)

            # ── 2. Burst: drive forward step by step ─────────────────
            self.get_logger().info("[LoRa] Driving burst toward tag...")
            last_rssi         = get_rssi() or current_rssi
            consecutive_drops = 0
            signal_dropped    = False

            for step in range(BURST_MAX_STEPS):
                self.cmd_vel_pub.publish(creep_fwd)
                time.sleep(STEP_TIME)
                self.cmd_vel_pub.publish(stop)
                time.sleep(0.2)

                rssi = get_rssi()
                if rssi is None:
                    continue

                self.get_logger().info(
                    f"[LoRa] Step RSSI={rssi} dBm | {rssi_description(rssi)}"
                )

                # ── 3. Confirm ────────────────────────────────────────
                if rssi >= RSSI_CONFIRM_THRESHOLD:
                    self.get_logger().info(
                        f"[LoRa] {expected_tag} CONFIRMED! RSSI={rssi}")
                    return True

                # ── 4. Stop if signal is dropping ─────────────────────
                if rssi <= last_rssi - 2:
                    consecutive_drops += 1
                    if consecutive_drops >= 2:
                        self.get_logger().warn(
                            f"[LoRa] Signal dropping ({last_rssi}→{rssi} dBm) "
                            f"— backing up and re-aligning..."
                        )
                        self.cmd_vel_pub.publish(creep_bwd)
                        time.sleep(0.5)
                        self.cmd_vel_pub.publish(stop)
                        current_rssi = get_rssi() or rssi
                        signal_dropped = True
                        break
                else:
                    consecutive_drops = 0
                    last_rssi  = rssi
                    current_rssi = rssi

        self.cmd_vel_pub.publish(stop)
        self.get_logger().warn(
            f"[LoRa] Gave up after {MAX_CYCLES} cycles. "
            f"Final RSSI={get_rssi()} dBm.")
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
                self.home_pose, expected_tag=None)
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