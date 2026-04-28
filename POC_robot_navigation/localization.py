import math


class Localizer:
    def __init__(self, intrinsics):

        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx_intr = intrinsics.ppx
        self.cy_intr = intrinsics.ppy

        self.prev_robot_pos = None

    def pixel_to_3d(self, u, v, depth):
        """
        Convert pixel + depth → real 3D world coordinates
        """

        if depth <= 0:
            return None

        Z = depth
        X = (u - self.cx_intr) * Z / self.fx
        Y = (v - self.cy_intr) * Z / self.fy

        return X, Y, Z

    def world_to_pixel(self, X, Z):

        pixel_x = int((X * self.fx / Z) + self.cx_intr)
        pixel_y = int((0 * self.fy / Z) + self.cy_intr)

        return pixel_x, pixel_y
    
    def compute_theta(self, prev, curr):

        if prev is None or curr is None:
            return 0.0

        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]

        if dx == 0 and dy == 0:
            return 0.0

        return math.atan2(dy, dx)

    def process(self, tracked_objects, depth_frame):

        robot_pose = None
        obstacles = []

        for obj in tracked_objects:

            x1, y1, x2, y2 = obj["bbox"]
            cls = obj["class"]
            track_id = obj.get("id", None)

            # 🔹 Center from tracker
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # 🔹 Get depth from RealSense
            depth = depth_frame.get_distance(cx, cy)

            coords = self.pixel_to_3d(cx, cy, depth)

            if coords is None:
                continue

            X, Y, Z = coords

            if cls == "robot":

                curr_pos = (X, Y)

                theta = self.compute_theta(self.prev_robot_pos, curr_pos)

                robot_pose = (
                    round(X, 2),
                    round(Y, 2),
                    round(Z, 2),
                    round(theta, 2)
                )

                self.prev_robot_pos = curr_pos

            else:
                obstacles.append({
                    "id": track_id,
                    "class": cls,
                    "pos": (
                        round(X, 2),
                        round(Y, 2),
                        round(Z, 2)
                    )
                })

        return robot_pose, obstacles