#!/usr/bin/env python3
import math, time, threading, subprocess, rclpy, rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
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
    def __init__(self):
        super().__init__("greedy_4_goals")
        self.map_frame  = "map"
        self.base_frame = "base_footprint"
        self.cb_group = ReentrantCallbackGroup()
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose", callback_group=self.cb_group)
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.goals = [
            self.make_goal(-2.50,  7.00, 0.0),
            self.make_goal(-3.58,  0.56, 0.0),
            self.make_goal( 3.73,  6.49, 0.0),
            self.make_goal(-2.50,  3.50, 0.0),
        ]
        self.get_logger().info("Waiting for Nav2 action server...")
        self.nav_client.wait_for_server()
        self.get_logger().info("Nav2 ready! Waiting 5s for full initialization...")
        time.sleep(5.0)
        self.get_logger().info("Starting greedy navigation!")
        self._nav_thread = threading.Thread(target=self.navigation_loop, daemon=True)
        self._nav_thread.start()

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
        return math.hypot(pose.pose.position.x - goal.pose.position.x, pose.pose.position.y - goal.pose.position.y)

    def clear_costmaps(self):
        self.get_logger().info(">>> Clearing costmaps...")
        for svc in ["/local_costmap/clear_entirely_local_costmap"]:
            try:
                result = subprocess.run(["ros2", "service", "call", svc, "nav2_msgs/srv/ClearEntireCostmap", "{}"], timeout=20, capture_output=True, text=True)
                if result.returncode == 0:
                    self.get_logger().info(f"    Cleared: {svc}")
                else:
                    self.get_logger().warn(f"    Failed: {result.stderr[:80]}")
            except Exception as e:
                self.get_logger().warn(f"    Error: {e}")
        self.get_logger().info(">>> Done clearing. Waiting 5s to settle...")
        time.sleep(5.0)

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

    def navigate_to(self, goal):
        attempt = 0
        while True:
            attempt += 1
            self.get_logger().info(f"Sending goal ({goal.pose.position.x:.2f},{goal.pose.position.y:.2f}) attempt {attempt}...")
            result = self.send_goal_once(goal)
            if result == 'succeeded':
                self.get_logger().info(f"Goal SUCCEEDED on attempt {attempt}!")
                return True
            elif result == 'rejected':
                self.get_logger().warn(f"Goal REJECTED (attempt {attempt}). Clearing costmaps and retrying...")
                self.clear_costmaps()
            elif result == 'failed':
                self.get_logger().warn(f"Navigation FAILED mid-route (attempt {attempt}). Clearing and retrying...")
                self.clear_costmaps()
            elif result == 'timeout':
                self.get_logger().warn(f"TIMEOUT on attempt {attempt}. Clearing and retrying...")
                self.clear_costmaps()

    def navigation_loop(self):
        self.get_logger().warn("Waiting for TF - set 2D Pose Estimate in RViz!")
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
            self.get_logger().info(f"===== GOAL {goal_num}/4 | Robot at ({rx:.2f},{ry:.2f}) =====")
            dists = [(self.euclidean(robot, g), i) for i, g in enumerate(self.goals)]
            dists.sort()
            for d, i in dists:
                self.get_logger().info(f"  #{i} ({self.goals[i].pose.position.x:.2f},{self.goals[i].pose.position.y:.2f}) dist={d:.2f}m")
            best_dist, best_idx = dists[0]
            goal = self.goals.pop(best_idx)
            self.get_logger().info(f"GREEDY PICK: ({goal.pose.position.x:.2f},{goal.pose.position.y:.2f}) | {len(self.goals)} goals remaining")
            self.clear_costmaps()
            ok = self.navigate_to(goal)
            if ok:
                robot_after = self.get_robot_pose()
                pos = f"({robot_after.pose.position.x:.2f},{robot_after.pose.position.y:.2f})" if robot_after else "unknown"
                self.get_logger().info(f"GOAL {goal_num} REACHED! Robot now at {pos}. {len(self.goals)} goals left.")
                self.get_logger().info("Waiting 10s for Nav2 to reset...")
                time.sleep(10.0)
                self.clear_costmaps()
            else:
                self.get_logger().warn("Requeueing failed goal.")
                self.goals.append(goal)
        self.get_logger().info("ALL GOALS COMPLETED!")

def main():
    rclpy.init()
    node = Greedy4Goals()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
