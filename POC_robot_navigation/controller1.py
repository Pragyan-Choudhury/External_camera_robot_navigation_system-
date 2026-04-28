import math


class PurePursuitController:
    """
    Pure Pursuit path tracking controller
    Generates:
        v → linear velocity (m/s)
        w → angular velocity (rad/s)
    """

    def __init__(self,
                 lookahead_distance=0.5,
                 max_linear_speed=0.5,
                 max_angular_speed=1.5):
        
        self.lookahead_distance = lookahead_distance
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed

    # ---------------------------------------------------------
    # Find Lookahead Target
    # ---------------------------------------------------------
    def find_lookahead_point(self, robot_x, robot_z, path):

        for px, pz in path:
            dist = math.hypot(px - robot_x, pz - robot_z)

            if dist >= self.lookahead_distance:
                return px, pz

        # If no point far enough → return last point
        return path[-1]

    # ---------------------------------------------------------
    # Main Control Law
    # ---------------------------------------------------------
    def compute_control(self, robot_pose, path):
        """
        robot_pose → (X, Y, Z, theta)
        path → list of (X, Z) waypoints

        Returns:
            v, w
        """

        if not path or robot_pose is None:
            return 0.0, 0.0

        robot_x, _, robot_z, theta = robot_pose

        # 1️⃣ Get lookahead target
        target_x, target_z = self.find_lookahead_point(
            robot_x, robot_z, path
        )

        # 2️⃣ Compute angle to target
        dx = target_x - robot_x
        dz = target_z - robot_z

        target_angle = math.atan2(dz, dx)

        # 3️⃣ Heading error
        alpha = target_angle - theta

        # Normalize angle
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # 4️⃣ Curvature
        L = self.lookahead_distance
        curvature = (2 * math.sin(alpha)) / L

        # 5️⃣ Compute velocities
        v = self.max_linear_speed
        w = v * curvature

        # Limit angular speed
        w = max(-self.max_angular_speed,
                min(self.max_angular_speed, w))

        # Slow down near goal
        goal_x, goal_z = path[-1]
        dist_to_goal = math.hypot(goal_x - robot_x,
                                  goal_z - robot_z)

        if dist_to_goal < 0.3:
            v = 0.0
            w = 0.0

        return v, w