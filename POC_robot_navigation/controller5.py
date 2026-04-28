import math


class PurePursuitController:

    def __init__(self,
                 max_linear_speed=0.4,
                 max_angular_speed=1.2,
                 goal_tolerance=0.3,
                 orientation_tolerance=0.1):

        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.goal_tolerance = goal_tolerance
        self.orientation_tolerance = orientation_tolerance

    # ---------------------------------------------------------
    def compute_control(self, robot_pose, path):

        if robot_pose is None or not path:
            return 0.0, 0.0

        robot_x, robot_y, _, theta = robot_pose
        goal_x, goal_y = path[-1]

        dx = goal_x - robot_x
        dy = goal_y - robot_y

        distance_to_goal = math.hypot(dx, dy)

        # --------------------------------------------------
        # PHASE 1 → MOVE STRAIGHT (NO ANGULAR VELOCITY)
        # --------------------------------------------------
        if distance_to_goal > self.goal_tolerance:

            v = self.max_linear_speed
            w = 0.0

            return v, w

        # --------------------------------------------------
        # PHASE 2 → ORIENT TOWARD GOAL DIRECTION
        # --------------------------------------------------
        target_angle = math.atan2(dy, dx)

        heading_error = target_angle - theta
        heading_error = math.atan2(
            math.sin(heading_error),
            math.cos(heading_error)
        )

        if abs(heading_error) > self.orientation_tolerance:

            k = 2.0
            w = -k * heading_error

            w = max(-self.max_angular_speed,
                    min(self.max_angular_speed, w))

            return 0.0, w

        # Final stop
        return 0.0, 0.0