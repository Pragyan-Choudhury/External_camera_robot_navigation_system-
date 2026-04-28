import cv2
import time
import numpy as np
import math

from realsense_cam import RealSenseCamera
from yolodetect_botrob import YOLODetector
from track import Tracker
from localization import Localizer
from map_builder import OccupancyGrid
from astar_planner import AStarPlanner  # <-- A* Planner
from controller1 import PurePursuitController


# 🔷 GLOBAL VARIABLES
clicked_goal = None
latest_tracked_objects = []
latest_depth_frame = None
localizer_ref = None


# 🔥 Mouse Click Callback
def mouse_callback(event, x, y, flags, param):
    global clicked_goal, latest_tracked_objects, latest_depth_frame, localizer_ref

    if event == cv2.EVENT_LBUTTONDOWN:

        print(f"[INFO] Mouse clicked at pixel ({x},{y})")

        for obj in latest_tracked_objects:

            x1, y1, x2, y2 = obj["bbox"]

            if x1 <= x <= x2 and y1 <= y <= y2:

                print(f"[INFO] Clicked on {obj['class']} ID:{obj['id']}")

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                depth = latest_depth_frame.get_distance(cx, cy)

                if depth == 0:
                    print("[WARNING] Invalid depth")
                    return

                X, Y, Z = localizer_ref.pixel_to_3d(cx, cy, depth)

                # Move goal slightly back toward robot
                Z = Z - 0.4   # 40cm before object

                clicked_goal = {
                    "id": obj["id"],
                    "class": obj["class"],
                    "pos": (round(X, 2), round(Y, 2), round(Z, 2))
                }

                print(f"[GOAL SET] {clicked_goal}")
                return


def main():

    global latest_tracked_objects
    global latest_depth_frame
    global localizer_ref
    global clicked_goal

    # 🔹 Camera
    camera = RealSenseCamera(width=640, height=480, fps=30)

    # 🔹 YOLO
    detector = YOLODetector(
        model_path="yolov8m.pt",
        conf=0.3,
        iou=0.5,
        debug=False
    )

    # 🔹 Tracker
    tracker = Tracker(iou_threshold=0.3, max_lost=10)

    # 🔹 Localizer
    localizer = Localizer(camera.intrinsics)
    localizer_ref = localizer
    
    intr = camera.intrinsics

    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy
    width_px = intr.width
    height_px = intr.height

    # Assume a certain depth (distance from camera to the floor or robot)
    depth_m = 6.0  # for example, 6 meters

    # Compute physical width and height at that depth
    width_m = width_px * depth_m / fx
    height_m = height_px * depth_m / fy
    # 🔹 Occupancy Grid (kept internally for future planning)
    #grid_map = OccupancyGrid(
    #    width=6.0,
    #    depth=6.0,
    #    resolution=0.1
    #)

    grid_map = OccupancyGrid(
    width=width_m,
    depth=height_m,
    resolution=0.1
    )
    # 🔹 A* Planner
    planner = AStarPlanner(grid_map)

    # 🔹 Controller
    controller = PurePursuitController(
        lookahead_distance=0.6,
        max_linear_speed=0.4,
        max_angular_speed=1.2
    )

    prev_time = 0

    print("[INFO] RealSense + YOLO + Tracker + Localization Running")

    cv2.namedWindow("Full Perception Pipeline")
    cv2.setMouseCallback("Full Perception Pipeline", mouse_callback)

    try:
        while True:

            color_frame, depth_frame = camera.get_frame()

            if color_frame is None:
                time.sleep(0.01)
                continue

            latest_depth_frame = depth_frame

            # 🔥 Detection
            detections = detector.detect(color_frame)

            # 🔥 Tracking
            tracked_objects = tracker.update(detections) if detections else []
            latest_tracked_objects = tracked_objects

            # 🔥 Localization
            robot_pose, obstacles = localizer.process(tracked_objects, depth_frame)

            # 🔥 Update Occupancy Grid (NO DISPLAY)
            grid_map.update(obstacles, inflation_radius=0.3)

            # 🔥 Compute A* Path if goal is set
            path = []
            if robot_pose and clicked_goal:
                #robot_pos = (robot_pose[0], robot_pose[2])    # X, Z
                robot_pos = (robot_pose[0], robot_pose[1])
                goal_pos = (clicked_goal["pos"][0], clicked_goal["pos"][2])

                print("[DEBUG] Robot Pos:", robot_pos)
                print("[DEBUG] Goal Pos:", goal_pos)

                path = planner.plan(robot_pos, goal_pos)

                print("[DEBUG] Path length:", len(path))
            
            # 🔹 Compute Control (v, w)
            v, w = 0.0, 0.0

            if robot_pose and path:
                v, w = controller.compute_control(robot_pose, path)

                print(f"[CONTROL] v={v:.2f} m/s  w={w:.2f} rad/s")

            # 🔥 Draw tracked objects
            for obj in tracked_objects:

                x1, y1, x2, y2 = obj["bbox"]
                cls = obj["class"]
                track_id = obj["id"]

                color = (0, 0, 255) if cls == "robot" else (0, 255, 0)

                cv2.rectangle(color_frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    color_frame,
                    f"{cls} ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
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

            # 🔥 Goal Display
            if clicked_goal:
                gx, gy, gz = clicked_goal["pos"]

                cv2.putText(
                    color_frame,
                    f"GOAL: {clicked_goal['pos']}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
            
            # 🔥 Display Control Output
            cv2.putText(
                color_frame,
                f"Linear Velocity v: {v:.2f} m/s",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
            
            cv2.putText(
                color_frame,
                f"Angular Velocity w: {w:.2f} rad/s",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # 🔹 Draw A* Path
            #for px, pz in path:
            #    gx, gy = grid_map.world_to_grid(px, pz)
                # scale to frame if needed (here we draw directly in pixels, simple visualization)
            #    cv2.circle(color_frame, (int(gx), int(gy)), 2, (255, 0, 0), -1)

            # for px, pz in path:

            #     img_x, img_y = localizer.world_to_pixel(px, pz)

            #     if 0 <= img_x < color_frame.shape[1] and 0 <= img_y < color_frame.shape[0]:
            #         cv2.circle(color_frame, (img_x, img_y), 4, (255, 0, 0), -1)

            # 🔹 Draw A* Path (Top-Down 2D Mapping)
            # if path and len(path) > 1:
            #     for i in range(len(path) - 1):

            #         x1_w, z1_w = path[i]
            #         x2_w, z2_w = path[i + 1]

            #         # Shift X from [-3,3] → [0,6]
            #         x1_shift = x1_w + (grid_map.width / 2)
            #         x2_shift = x2_w + (grid_map.width / 2)

            #         # Z is already 0 → 6
            #         #z1_shift = z1_w
            #         #z2_shift = z2_w
            #         z1_flip = grid_map.depth - z1_w
            #         z2_flip = grid_map.depth - z2_w

            #         # Scale to image resolution
            #         x1_img = int((x1_shift / grid_map.width) * width_px)
            #         # y1_img = int((z1_shift / grid_map.depth) * height_px)
            #         y1_img = int((z1_flip / grid_map.depth) * height_px)

            #         x2_img = int((x2_shift / grid_map.width) * width_px)
            #         # y2_img = int((z2_shift / grid_map.depth) * height_px)
            #         y2_img = int((z2_flip / grid_map.depth) * height_px)

            #         cv2.line(color_frame, (x1_img, y1_img),
            #                               (x2_img, y2_img),
            #                               (255, 0, 255), 2)
                    
            # 🔹 FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            cv2.putText(
                color_frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # 🔥 SINGLE WINDOW DISPLAY
            cv2.imshow("Full Perception Pipeline", color_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("[INFO] Program Terminated")


if __name__ == "__main__":
    main()