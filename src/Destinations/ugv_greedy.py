#!/usr/bin/env python3
import csv, os, math, time, threading, subprocess, rclpy, rclpy.time
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
MAX_RECOVERIES    = 10
CHEAPER_THRESHOLD = 0.75

# ------------------------------------------------------------------
# LoRa / RSSI constants
# ------------------------------------------------------------------
RSSI_CONFIRM_THRESHOLD = -60
RSSI_EXCELLENT         = -50
RSSI_GOOD              = -70
RSSI_WEAK              = -90
RSSI_VERY_WEAK         = -100
CREEP_SPEED            = 0.08
CREEP_CHECK_INTERVAL   = 0.5

# ------------------------------------------------------------------
# ── Logging configuration ─────────────────────────────────────────
# File 1: one row every 5 s during Nav2 navigation (x,y,t per segment)
# File 2: one row per waypoint with the final arrival result after LoRa
# ------------------------------------------------------------------
LOG_DIR       = os.path.expanduser("~/robot_logs")
NAV_LOG_FILE  = os.path.join(LOG_DIR, "nav_trajectory.csv")
LORA_LOG_FILE = os.path.join(LOG_DIR, "lora_arrivals.csv")
NAV_LOG_INTERVAL = 5.0   # seconds between pose samples in file 1

# ------------------------------------------------------------------
# Goal definitions
# ------------------------------------------------------------------
GOAL_DEFS = [
    {"x": 1.43,   "y":  0.168, "yaw": 0.000866, "tag": "TAG_001"},
    {"x": 1.22,  "y": -1.72,  "yaw": 0.0021, "tag": "TAG_002"},
    {"x": 0.60, "y": -1.70, "yaw": 0.00252, "tag": "TAG_003"},
    {"x": 0.658, "y": -0.3730, "yaw": 0.00391, "tag": "TAG_004"},
]


def rssi_description(rssi):
    if rssi is None:                       return "No signal"
    elif rssi >= RSSI_EXCELLENT:           return "Excellent"
    elif rssi >= RSSI_GOOD:               return "Good"
    elif rssi >= RSSI_CONFIRM_THRESHOLD:  return "OK - above threshold"
    elif rssi >= RSSI_WEAK:              return "Weak - creeping"
    elif rssi >= RSSI_VERY_WEAK:         return "Very Weak"
    else:                                 return "No signal / noise"


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
        self._lora_tag_id    = None
        self._lora_rssi      = -999
        self._lora_lock      = threading.Lock()
        self._lora_last_seen = {}

        self.create_subscription(
            String, '/lora_tag',
            self._lora_callback, 10,
            callback_group=self.cb_group
        )
        self.get_logger().info("LoRa subscriber ready on /lora_tag")

        # ── Goals ──────────────────────────────────────────────────
        self.goals = [
            (self.make_goal(d["x"], d["y"], d["yaw"]), d["tag"])
            for d in GOAL_DEFS
        ]

        # ── Trajectory logging ──────────────────────────────────────
        # _from_label tracks where the robot currently is (for segment names)
        self._from_label     = "HOME"
        self._nav_log_stop   = threading.Event()
        self._nav_log_thread = None
        self._lora_file_lock      = threading.Lock()  # protects direct appends to File 2
        self._mission_start_time  = None              # set once on first nav goal, never reset
        self._init_log_files()
        self.get_logger().info(
            f"Logging nav to:  {NAV_LOG_FILE}\n"
            f"Logging LoRa to: {LORA_LOG_FILE}"
        )

        self.get_logger().info("Waiting for Nav2 action server...")
        self.nav_client.wait_for_server()
        self.plan_client.wait_for_server()
        self.get_logger().info("Nav2 ready! Waiting 15s for full initialization...")
        time.sleep(15.0)
        self.get_logger().info("Starting greedy navigation!")

        self._nav_thread = threading.Thread(
            target=self.navigation_loop, daemon=True)
        self._nav_thread.start()

    # ==================================================================
    # ── Logging helpers ───────────────────────────────────────────────
    # ==================================================================

    def _init_log_files(self):
        """Create log directory and write CSV headers (once per run)."""
        os.makedirs(LOG_DIR, exist_ok=True)

        # Shared header builder — File 2 extends File 1 with three extra columns
        nav_header  = ['segment', 'elapsed_s', 'ros_time_s', 'x', 'y',
                        'goal_x', 'goal_y', 'goal_tag']
        lora_header = nav_header + ['phase', 'rssi_dbm', 'lora_confirmed']
        # phase values used in File 2:
        #   "nav2"   — pose sample every 5 s while Nav2 is driving
        #   "lora"   — pose + RSSI sample during LoRa approach steps
        #   "result" — final row: outcome of the LoRa confirmation attempt

        for path, header in [(NAV_LOG_FILE, nav_header),
                              (LORA_LOG_FILE, lora_header)]:
            if not os.path.exists(path):
                with open(path, 'w', newline='') as f:
                    csv.writer(f).writerow(header)
            else:
                # Blank separator line between runs so MATLAB can tell them apart
                with open(path, 'a', newline='') as f:
                    csv.writer(f).writerow([])

    # ------------------------------------------------------------------
    # File 1 + File 2 nav-phase logger (background thread)
    # Writes to File 1 (nav_trajectory.csv) every NAV_LOG_INTERVAL seconds.
    # Simultaneously writes the same row to File 2 (lora_arrivals.csv) with
    # phase="nav2" and blank rssi_dbm / lora_confirmed columns.
    # ------------------------------------------------------------------

    def _nav_log_worker(self, segment: str, goal: PoseStamped, goal_tag: str,
                        seg_start: float):
        """Background thread: records robot pose every NAV_LOG_INTERVAL seconds."""
        with open(NAV_LOG_FILE,  'a', newline='') as f1, \
             open(LORA_LOG_FILE, 'a', newline='') as f2:
            w1 = csv.writer(f1)
            w2 = csv.writer(f2)
            while not self._nav_log_stop.is_set():
                pose    = self.get_robot_pose()
                elapsed = round(time.time() - seg_start, 2)
                ros_t   = round(self.get_clock().now().nanoseconds / 1e9, 3)
                if pose is not None:
                    rx = round(pose.pose.position.x, 4)
                    ry = round(pose.pose.position.y, 4)
                    gx = round(goal.pose.position.x, 4)
                    gy = round(goal.pose.position.y, 4)
                    base_row = [segment, elapsed, ros_t, rx, ry, gx, gy, goal_tag]
                    w1.writerow(base_row)
                    w2.writerow(base_row + ['nav2', '', ''])
                    f1.flush()
                    f2.flush()
                self._nav_log_stop.wait(timeout=NAV_LOG_INTERVAL)

    def _start_nav_logging(self, segment: str, goal: PoseStamped, goal_tag: str,
                           seg_start: float):
        self._nav_log_stop.clear()
        self._nav_log_thread = threading.Thread(
            target=self._nav_log_worker,
            args=(segment, goal, goal_tag, seg_start),
            daemon=True,
            name="nav_logger",
        )
        self._nav_log_thread.start()
        self.get_logger().info(f"[LOG] Nav logging started — segment: {segment}")

    def _stop_nav_logging(self):
        self._nav_log_stop.set()
        if self._nav_log_thread is not None:
            self._nav_log_thread.join(timeout=3.0)
            self._nav_log_thread = None
        self.get_logger().info("[LOG] Nav logging stopped.")

    # ------------------------------------------------------------------
    # File 2 — direct row append (called during LoRa phase and for result)
    #
    # phase="lora"   : one call per RSSI sample during the approach
    # phase="result" : one call at the very end with confirmed=0/1
    # ------------------------------------------------------------------

    def _append_lora_file_row(
        self,
        segment:   str,
        seg_start: float,
        goal:      PoseStamped,
        goal_tag:  str,
        phase:     str,        # "lora" | "result"
        rssi,                  # int | None
        confirmed,             # '' during lora steps, 0/1 for result row
    ):
        pose    = self.get_robot_pose()
        elapsed = round(time.time() - seg_start, 2)
        ros_t   = round(self.get_clock().now().nanoseconds / 1e9, 3)

        if pose is not None:
            rx = round(pose.pose.position.x, 4)
            ry = round(pose.pose.position.y, 4)
        else:
            rx = ry = ''

        gx = round(goal.pose.position.x, 4)
        gy = round(goal.pose.position.y, 4)

        with self._lora_file_lock:
            with open(LORA_LOG_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([
                    segment, elapsed, ros_t,
                    rx, ry, gx, gy, goal_tag,
                    phase,
                    rssi if rssi is not None else '',
                    confirmed,
                ])

    # ==================================================================
    # ── Existing node methods (unchanged except where noted) ──────────
    # ==================================================================

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

    def stop_robot(self):
        self._shutdown = True
        self._stop_nav_logging()   # make sure logger is killed on shutdown
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

    def score_goal(self, robot, goal):
        dist = self.euclidean(robot, goal)
        if dist < self.OBSTACLE_REPLAN_RADIUS:
            cost = self.get_path_cost(robot, goal)
            return cost if cost is not None else float('inf')
        return dist

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

    def trigger_amcl_relocalize(self):
        pose = self.get_robot_pose()
        if pose is None:
            return
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id   = "map"
        msg.header.stamp      = self.get_clock().now().to_msg()
        msg.pose.pose         = pose.pose
        msg.pose.covariance[0]  = 0.25
        msg.pose.covariance[7]  = 0.25
        msg.pose.covariance[35] = 0.1
        self.initial_pose_pub.publish(msg)
        self.get_logger().info("AMCL relocalization triggered")

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

    def _mid_drive_check(self, current_goal, is_home=False):
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
        if is_home:
            return 'continue'

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

    def _drive_with_monitoring(self, goal, is_home=False):
        gh, result_event, res_box = self._send_goal_async(goal)

        if gh is None:
            self.get_logger().warn("Goal rejected or send timed out.")
            return 'failed'

        self.get_logger().info("Goal accepted! Monitoring mid-drive...")

        while not result_event.wait(timeout=CHECK_INTERVAL):
            decision = self._mid_drive_check(goal, is_home=is_home)
            if decision != 'continue':
                self._cancel_current_goal(reason=decision)
                return 'failed' if is_home else 'switch_goal'

        res = res_box[0]
        if res is not None and res.status == GoalStatus.STATUS_SUCCEEDED:
            return 'succeeded'
        return 'failed'

    # ------------------------------------------------------------------
    # LoRa approach
    # CHANGED: now returns (confirmed: bool, final_rssi: int | None)
    # so the caller can log the RSSI into file 2.
    # ------------------------------------------------------------------

    def _approach_via_lora(self, expected_tag: str, lora_log_fn=None):
        """
        Returns (confirmed: bool, final_rssi: int | None)
        lora_log_fn(rssi) — optional callback that writes one row to File 2
        with phase="lora" and the current robot pose + rssi.
        """
        def _log(rssi):
            if lora_log_fn is not None:
                lora_log_fn(rssi)

        stop       = Twist()
        creep_fwd  = Twist(); creep_fwd.linear.x  =  CREEP_SPEED
        creep_bwd  = Twist(); creep_bwd.linear.x  = -CREEP_SPEED
        rotate_cw  = Twist(); rotate_cw.angular.z  = -0.5
        rotate_ccw = Twist(); rotate_ccw.angular.z =  0.5

        def get_rssi():
            with self._lora_lock:
                entry = self._lora_last_seen.get(expected_tag)
                if entry and (time.time() - entry[1]) < 5.0:
                    return entry[0]
            return None

        def sample_rssi(duration=1.0):
            samples, deadline = [], time.time() + duration
            while time.time() < deadline:
                r = get_rssi()
                if r is not None:
                    samples.append(r)
                time.sleep(0.2)
            return sum(samples) / len(samples) if samples else None

        for _ in range(10):
            self.cmd_vel_pub.publish(stop)
            time.sleep(0.1)
        time.sleep(1.5)

        rssi = get_rssi()
        self.get_logger().info(
            f"[LoRa] Arrived at {expected_tag}. "
            f"RSSI={rssi} dBm | {rssi_description(rssi)}"
        )
        _log(rssi)    # Phase 1 arrival check

        if rssi is None:
            self.get_logger().warn(f"[LoRa] {expected_tag} — no signal.")
            return False, None

        if rssi >= RSSI_CONFIRM_THRESHOLD:
            self.get_logger().info(f"[LoRa] {expected_tag} CONFIRMED! RSSI={rssi}")
            return True, rssi

        self.get_logger().warn(
            f"[LoRa] Weak signal ({rssi} dBm). Rotating to find best heading...")

        COARSE_TIME    = 3.0
        FINE_TIME      = 8.0

        self.cmd_vel_pub.publish(rotate_ccw)
        time.sleep(COARSE_TIME)
        self.cmd_vel_pub.publish(stop); time.sleep(0.3)
        rssi_ccw = sample_rssi(0.5)
        self.get_logger().info(f"[LoRa] Coarse CCW: RSSI={rssi_ccw} dBm")
        _log(rssi_ccw)

        self.cmd_vel_pub.publish(rotate_cw)
        time.sleep(COARSE_TIME * 2)
        self.cmd_vel_pub.publish(stop); time.sleep(0.3)
        rssi_cw = sample_rssi(0.5)
        self.get_logger().info(f"[LoRa] Coarse CW:  RSSI={rssi_cw} dBm")
        _log(rssi_cw)

        self.cmd_vel_pub.publish(rotate_ccw)
        time.sleep(COARSE_TIME)
        self.cmd_vel_pub.publish(stop); time.sleep(0.3)

        if rssi_ccw is not None and (rssi_cw is None or rssi_ccw >= rssi_cw):
            fine_dir, fine_label, best_rssi = rotate_ccw, "CCW", rssi_ccw
        else:
            fine_dir, fine_label, best_rssi = rotate_cw, "CW", rssi_cw

        best_rssi = max(r for r in [rssi, best_rssi] if r is not None)
        self.get_logger().info(
            f"[LoRa] Chosen side: {fine_label} "
            f"(CCW={rssi_ccw} dBm, CW={rssi_cw} dBm)"
        )

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
                _log(r)    # fine scan sample
                if r >= RSSI_CONFIRM_THRESHOLD:
                    self.cmd_vel_pub.publish(stop)
                    self.get_logger().info(
                        f"[LoRa] {expected_tag} CONFIRMED during fine scan! RSSI={r}")
                    return True, r
                if r > best_rssi:
                    best_rssi        = r
                    drops_after_peak = 0
                elif best_rssi > rssi and r <= best_rssi - 2:
                    drops_after_peak += 1
                    if drops_after_peak >= 2:
                        self.get_logger().info(
                            f"[LoRa] Peak at t={elapsed:.1f}s, "
                            f"peak={best_rssi} dBm, now={r} dBm")
                        break
                else:
                    drops_after_peak = 0
            time.sleep(0.3)

        self.cmd_vel_pub.publish(stop); time.sleep(0.3)
        self.get_logger().info(
            f"[LoRa] Best heading found. Peak RSSI={best_rssi} dBm")

        rssi_at_heading = sample_rssi(1.0)
        self.get_logger().info(
            f"[LoRa] At best heading. RSSI={rssi_at_heading} dBm")
        _log(rssi_at_heading)    # best-heading sample

        if rssi_at_heading and rssi_at_heading >= RSSI_CONFIRM_THRESHOLD:
            self.get_logger().info(f"[LoRa] {expected_tag} CONFIRMED at best heading!")
            return True, rssi_at_heading

        ALIGN_TIME      = 3.0
        STEP_TIME       = 0.5
        BURST_MAX_STEPS = 5
        MAX_CYCLES      = 5

        current_rssi = rssi_at_heading or best_rssi

        for cycle in range(MAX_CYCLES):
            self.get_logger().info(
                f"[LoRa] Cycle {cycle+1}/{MAX_CYCLES} — aligning heading...")

            self.cmd_vel_pub.publish(rotate_ccw)
            time.sleep(ALIGN_TIME)
            self.cmd_vel_pub.publish(stop); time.sleep(0.2)
            probe_ccw = sample_rssi(0.3)
            _log(probe_ccw)    # cycle CCW probe

            self.cmd_vel_pub.publish(rotate_cw)
            time.sleep(ALIGN_TIME * 2)
            self.cmd_vel_pub.publish(stop); time.sleep(0.2)
            probe_cw = sample_rssi(0.3)
            _log(probe_cw)     # cycle CW probe

            self.cmd_vel_pub.publish(rotate_ccw)
            time.sleep(ALIGN_TIME)
            self.cmd_vel_pub.publish(stop); time.sleep(0.2)

            self.get_logger().info(
                f"[LoRa] Probe — CCW={probe_ccw} dBm, CW={probe_cw} dBm, "
                f"centre={current_rssi} dBm"
            )

            for p in [probe_ccw, probe_cw]:
                if p is not None and p >= RSSI_CONFIRM_THRESHOLD:
                    self.get_logger().info(
                        f"[LoRa] {expected_tag} CONFIRMED during alignment! RSSI={p}")
                    return True, p

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

            self.get_logger().info("[LoRa] Driving burst toward tag...")
            last_rssi         = get_rssi() or current_rssi
            consecutive_drops = 0
            signal_dropped    = False

            for step in range(BURST_MAX_STEPS):
                self.cmd_vel_pub.publish(creep_fwd)
                time.sleep(STEP_TIME)
                self.cmd_vel_pub.publish(stop); time.sleep(0.2)

                rssi = get_rssi()
                if rssi is None:
                    continue

                self.get_logger().info(
                    f"[LoRa] Step RSSI={rssi} dBm | {rssi_description(rssi)}"
                )
                _log(rssi)    # burst creep step

                if rssi >= RSSI_CONFIRM_THRESHOLD:
                    self.get_logger().info(
                        f"[LoRa] {expected_tag} CONFIRMED! RSSI={rssi}")
                    return True, rssi

                if rssi <= last_rssi - 2:
                    consecutive_drops += 1
                    if consecutive_drops >= 2:
                        self.get_logger().warn(
                            f"[LoRa] Signal dropping ({last_rssi}→{rssi} dBm) "
                            f"— backing up and re-aligning...")
                        self.cmd_vel_pub.publish(creep_bwd)
                        time.sleep(0.5)
                        self.cmd_vel_pub.publish(stop)
                        current_rssi   = get_rssi() or rssi
                        _log(current_rssi)    # post-backup position
                        signal_dropped = True
                        break
                else:
                    consecutive_drops = 0
                    last_rssi    = rssi
                    current_rssi = rssi

        self.cmd_vel_pub.publish(stop)
        final_rssi_val = get_rssi()
        self.get_logger().warn(
            f"[LoRa] Gave up after {MAX_CYCLES} cycles. "
            f"Final RSSI={final_rssi_val} dBm.")
        return False, final_rssi_val

    # ------------------------------------------------------------------
    # Navigate with replan
    # CHANGED: accepts `segment` for logging; integrates start/stop logging
    #          and calls _log_lora_arrival once per succeeded goal.
    # ------------------------------------------------------------------

    def navigate_with_replan(
        self, goal, expected_tag=None, max_attempts=None,
        is_home=False, segment="UNKNOWN"
    ):
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

            # ── Start logging pose every 5 s to File 1 and File 2 (nav2 phase)
            if self._mission_start_time is None:
                self._mission_start_time = time.time()   # set once, never reset
            self._start_nav_logging(segment, goal, expected_tag or 'HOME',
                                    self._mission_start_time)
            result = self._drive_with_monitoring(goal, is_home=is_home)
            self._stop_nav_logging()
            # ──────────────────────────────────────────────────────────────

            if result == 'succeeded':
                self.get_logger().info(f"Goal SUCCEEDED on attempt {attempt}!")

                # Callback used inside _approach_via_lora to write LoRa-phase rows
                def lora_log_fn(rssi, _seg=segment,
                                _goal=goal, _tag=expected_tag or 'HOME'):
                    self._append_lora_file_row(
                        _seg, self._mission_start_time, _goal, _tag, 'lora', rssi, '')

                final_rssi = None
                if expected_tag is not None:
                    with self._lora_lock:
                        entry    = self._lora_last_seen.get(expected_tag)
                        pre_rssi = entry[0] if entry and (time.time() - entry[1]) < 5.0 else None

                    if pre_rssi is None:
                        self.get_logger().info(
                            f"[RESULT] {expected_tag} — "
                            f"NAV2 COORDINATES ONLY (no LoRa signal)."
                        )
                        lora_confirmed = False
                    else:
                        lora_confirmed, final_rssi = self._approach_via_lora(
                            expected_tag, lora_log_fn=lora_log_fn)
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
                else:
                    lora_confirmed = True   # home return always counts as success

                # ── Write the closing result row to File 2 ─────────────
                self._append_lora_file_row(
                    segment, self._mission_start_time, goal, expected_tag or 'HOME',
                    'result', final_rssi, int(lora_confirmed)
                )
                self.get_logger().info(
                    f"[LOG] Result row written — segment={segment} "
                    f"confirmed={lora_confirmed} rssi={final_rssi}"
                )

                return True, lora_confirmed

            elif result == 'switch_goal':
                return False, False

            elif result == 'failed':
                if is_home:
                    self.get_logger().warn(
                        f"Returning home — clearing and retrying.")
                    self.clear_costmaps()
                    continue
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
    # CHANGED: builds segment label (e.g. "HOME→TAG_001") and passes it
    #          to navigate_with_replan; updates _from_label after success.
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

            # ── Build segment label for this leg ──────────────────
            segment = f"{self._from_label}→{expected_tag}"

            self.get_logger().info(
                f"GREEDY PICK: "
                f"({goal.pose.position.x:.2f},{goal.pose.position.y:.2f}) | "
                f"Expecting: {expected_tag} | segment: {segment} | "
                f"{len(self.goals)} goals remaining after this"
            )

            with self._lora_lock:
                self._lora_tag_id = None
                self._lora_rssi   = -999
                self._lora_last_seen.pop(expected_tag, None)

            self.clear_costmaps()
            ok, lora_confirmed = self.navigate_with_replan(
                goal, expected_tag=expected_tag, segment=segment)

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

                # ── Advance from_label for next segment ───────────
                self._from_label = expected_tag

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

            # ── Final segment: last tag → HOME ────────────────────
            home_segment = f"{self._from_label}→HOME"
            ok, _ = self.navigate_with_replan(
                self.home_pose, expected_tag=None,
                max_attempts=None, is_home=True,
                segment=home_segment
            )
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