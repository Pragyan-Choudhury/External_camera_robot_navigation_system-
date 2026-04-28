import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import cv2
import numpy as np
import math

from realsense_cam import RealSenseCamera
#from yolodetect_botrob import YOLODetector
from yolodetect_go2 import YOLODetector
from track import Tracker
from localization import Localizer
from map_builder import OccupancyGrid
from astar_planner import AStarPlanner
from controller5 import PurePursuitController


class FullPipelineNavigator(Node):

    def __init__(self):
        super().__init__('full_pipeline_nav')

        # ---------------- ROS2 Publisher ----------------
        self.cmd_pub = self.create_publisher(
            Twist,
        #    '/model/vehicle_blue/cmd_vel',
            '/cmd_vel',
            10
        )

        # ---------------- Odometry ----------------
        self.odom_sub = self.create_subscription(
            Odometry,
        #    '/model/vehicle_blue/odometry',
            '/odom',
            self.odom_callback,
            10
        )

        self.current_pose = None

        # ---------------- Camera ----------------
        self.camera = RealSenseCamera(width=640, height=480, fps=30)

        # ---------------- Perception ----------------
        #self.detector = YOLODetector(
        #    model_path="yolov8m.pt",
        #    conf=0.3,
        #    iou=0.5,
        #    debug=False
        #)
        self.detector = YOLODetector(
            object_model_path="yolov8m.pt",
            robot_model_path="best.pt",
            conf=0.3,
            iou=0.5,
            debug=False
        )

        self.tracker = Tracker(iou_threshold=0.3, max_lost=10)
        self.localizer = Localizer(self.camera.intrinsics)

        # ---------------- Map ----------------
        intr = self.camera.intrinsics
        depth_m = 6.0

        width_m = intr.width * depth_m / intr.fx
        height_m = intr.height * depth_m / intr.fy

        self.grid_map = OccupancyGrid(
            width=width_m,
            depth=height_m,
            resolution=0.1
        )

        self.planner = AStarPlanner(self.grid_map)

        # ---------------- Controller ----------------
        #self.controller = PurePursuitController(
        #    lookahead_distance=0.6,
        #    max_linear_speed=0.4,
        #    max_angular_speed=1.2
        #)

        self.controller = PurePursuitController(
            max_linear_speed=0.4,
            max_angular_speed=1.2,
            goal_tolerance=0.3,
            orientation_tolerance=0.1
        )
        # ---------------- State ----------------
        self.clicked_goal = None
        self.latest_tracked_objects = []
        self.latest_depth_frame = None

        # ---------------- OpenCV ----------------
        cv2.namedWindow("Full Perception Pipeline")
        cv2.setMouseCallback("Full Perception Pipeline", self.mouse_callback)

        # ---------------- Loop ----------------
        self.timer = self.create_timer(0.05, self.process_loop)

        print("🚀 Full Pipeline Navigator Started")

    # =====================================================
    # ODOMETRY
    # =====================================================
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        yaw = self.get_yaw_from_quaternion(q)

        self.current_pose = (x, y, 0.0, yaw)

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # =====================================================
    # MOUSE CLICK GOAL
    # =====================================================
    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            for obj in self.latest_tracked_objects:

                x1, y1, x2, y2 = obj["bbox"]

                if x1 <= x <= x2 and y1 <= y <= y2:

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    depth = self.latest_depth_frame.get_distance(cx, cy)

                    if depth == 0:
                        return

                    X, Y, Z = self.localizer.pixel_to_3d(cx, cy, depth)

                    #self.clicked_goal = (X, Z - 0.4)
                    self.clicked_goal = {
                        "id": obj["id"],
                        "pos": (X, Z - 0.4)
                    }
                    print(f"[GOAL SET] {self.clicked_goal}")

    # =====================================================
    # MAIN LOOP
    # =====================================================
    def process_loop(self):

        color_frame, depth_frame = self.camera.get_frame()

        if color_frame is None:
            return

        self.latest_depth_frame = depth_frame

        # ---------------- Detection ----------------
        detections = self.detector.detect(color_frame)

        # ---------------- Tracking ----------------
        tracked_objects = self.tracker.update(detections) if detections else []
        self.latest_tracked_objects = tracked_objects

        # ---------------- Localization ----------------
        robot_pose, obstacles = self.localizer.process(tracked_objects, depth_frame)
        self.grid_map.update(obstacles, inflation_radius=0.3)

        # ---------------- Planning ----------------
        path = []

        if self.current_pose and self.clicked_goal:
            #robot_pos = (self.current_pose[0], self.current_pose[2])
            robot_pos = (self.current_pose[0], self.current_pose[1])
            goal_pos = self.clicked_goal
            

            print("[DEBUG] Robot Pos:", robot_pos)
            print("[DEBUG] Goal Pos:", goal_pos)

            robot_detected = any(obj["class"] == "robot" for obj in tracked_objects)

            if not robot_detected:
                self.clicked_goal = None
                print("[INFO] Robot not detected — stopping control")
                return
            
            
            #path = self.planner.plan(robot_pos, self.clicked_goal)
            goal_pos = self.clicked_goal["pos"]
            path = self.planner.plan(robot_pos, goal_pos)

            print("[DEBUG] Path length:", len(path))
        # ---------------- Control ----------------
        v, w = 0.0, 0.0

        #if self.current_pose and path:
        #    v, w = self.controller.compute_control(self.current_pose, path)
        
        if self.current_pose and len(path) > 1:
            v, w = self.controller.compute_control(self.current_pose, path)

            print(f"[CONTROL] v={v:.2f} m/s  w={w:.2f} rad/s")
        else:
            v, w = 0.0, 0.0


            print(f"[CONTROL] v={v:.2f} m/s  w={w:.2f} rad/s")

        # ---------------- Publish ----------------
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

        # =====================================================
        # 🔥 VISUALIZATION FIX (BOUNDING BOX DRAWING)
        # =====================================================
        for obj in tracked_objects:

            x1, y1, x2, y2 = map(int, obj["bbox"])
            cls = obj["class"]
            track_id = obj["id"]

            color = (0, 0, 255) if cls == "robot" else (0, 255, 0)

            cv2.rectangle(color_frame, (x1, y1), (x2, y2), color, 2)

            #cv2.rectangle(
            #    color_frame,
            #    (x1, y1),
            #    (x2, y2),
            #    (0, 255, 0),
            #    2
            #)
            cv2.putText(
                color_frame,
                f"{cls} ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            #cv2.putText(
            #    color_frame,
            #    f"ID:{obj_id}",
            #    (x1, y1 - 5),
            #    cv2.FONT_HERSHEY_SIMPLEX,
            #    0.5,
            #    (0, 255, 0),
            #    2
            #)
        
        # ---------------- GOAL DISPLAY ----------------
        if self.clicked_goal:

            goal_id = self.clicked_goal["id"]
            gx, gz = self.clicked_goal["pos"]

            cv2.putText(
                color_frame,
                f"GOAL ID:{goal_id}  X:{gx:.2f}  Z:{gz:.2f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # 🔥 Robot Pose Display
        if robot_pose:
            X, Y, Z, theta = robot_pose

            cv2.putText(
                color_frame,
                f"Robot: X={X:.2f}  Y={Y:.2f}  Z={Z:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        # ---------------- UI INFO ----------------
        cv2.putText(color_frame, f"v: {v:.2f}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(color_frame, f"w: {w:.2f}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ---------------- SHOW ----------------
        cv2.imshow("Full Perception Pipeline", color_frame)
        cv2.waitKey(1)

    # =====================================================
    def destroy_node(self):
        self.camera.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


# =========================================================
# MAIN
# =========================================================
def main():
    rclpy.init()
    node = FullPipelineNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()