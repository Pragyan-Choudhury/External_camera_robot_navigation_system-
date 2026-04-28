import heapq
import math


class AStarPlanner:
    """
    A* path planner for 2D occupancy grids.
    """
    def __init__(self, occupancy_grid):
        """
        occupancy_grid → instance of OccupancyGrid
        """
        self.grid_map = occupancy_grid
        self.grid = occupancy_grid.grid
        self.rows = occupancy_grid.rows
        self.cols = occupancy_grid.cols

        # 8-connected grid moves: (dx, dy, cost)
        self.moves = [
            (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),    # 4-connectivity
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),   # diagonals
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))
        ]

    def heuristic(self, node, goal):
        """
        Euclidean distance as heuristic
        """
        x1, y1 = node
        x2, y2 = goal
        return math.hypot(x2 - x1, y2 - y1)

    def plan(self, start_pos, goal_pos):
        """
        start_pos / goal_pos → tuple (X, Z) in meters
        Returns list of path points in meters [(X1, Z1), ...]
        """

        # Convert world coordinates to grid
        start = self.grid_map.world_to_grid(*start_pos)
        goal = self.grid_map.world_to_grid(*goal_pos)

        # Priority queue for open set
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start, None))

        came_from = {}  # child → parent
        cost_so_far = {start: 0}

        while open_set:
            _, current_cost, current, parent = heapq.heappop(open_set)

            if current == goal:
                came_from[current] = parent
                break

            if current not in came_from:
                came_from[current] = parent

            cx, cy = current

            for dx, dy, move_cost in self.moves:
                nx, ny = cx + dx, cy + dy

                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if self.grid[ny][nx] == 1:  # obstacle
                        continue

                    new_cost = current_cost + move_cost
                    neighbor = (nx, ny)

                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (priority, new_cost, neighbor, current))

        # Reconstruct path
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = came_from.get(node, None)
        path.reverse()

        # Convert path back to world coordinates
        world_path = [self.grid_map.grid_to_world(x, y) for x, y in path]
        return world_path


# Optional: Add this method in OccupancyGrid to convert grid → world
#def grid_to_world(self, gx, gy):
#    """
#    Convert grid coordinates → world (X, Z)
#    """
#    X = (gx - self.origin_x) * self.resolution
#    Z = (gy - self.origin_z) * self.resolution
#    return X, Z


# Attach method to OccupancyGrid class dynamically
#OccupancyGrid.grid_to_world = grid_to_world