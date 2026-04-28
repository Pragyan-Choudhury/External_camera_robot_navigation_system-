import numpy as np


class OccupancyGrid:
    def __init__(self, width=6.0, depth=6.0, resolution=0.1):
        """
        width  → left-right size in meters (X axis)
        depth  → forward size in meters (Z axis)
        resolution → size of each grid cell in meters
        """

        self.width = width
        self.depth = depth
        self.resolution = resolution

        # Number of grid cells
        self.cols = int(width / resolution)
        self.rows = int(depth / resolution)

        # Create empty grid
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # Origin shift (robot placed at bottom-center of grid)
        self.origin_x = self.cols // 2   # center horizontally
        self.origin_z = 0                # robot at bottom

        print(f"[INFO] Occupancy Grid Initialized: {self.rows}x{self.cols}")

    # ----------------------------------------------------
    # Convert world (X,Z) → grid coordinates
    # ----------------------------------------------------
    def world_to_grid(self, X, Z):

        gx = int((X / self.resolution) + self.origin_x)
        gy = int((Z / self.resolution) + self.origin_z)

        return gx, gy

    # ----------------------------------------------------
    # Convert grid → world coordinates
    # ----------------------------------------------------
    def grid_to_world(self, gx, gy):
        """
        Convert grid coordinates → world (X, Z)
        """
        X = (gx - self.origin_x) * self.resolution
        Z = (gy - self.origin_z) * self.resolution
        return X, Z
    
    # ----------------------------------------------------
    # Update grid with obstacles
    # ----------------------------------------------------
    def update(self, obstacles, inflation_radius=0.2):

        # Reset grid
        self.grid.fill(0)

        # Convert inflation radius (meters → cells)
        inflation_cells = int(inflation_radius / self.resolution)

        for obs in obstacles:

            # Expecting pos = (X, Y, Z)
            X, Y, Z = obs["pos"]

            # Ignore invalid depth
            if Z <= 0:
                continue

            # Convert to grid
            gx, gy = self.world_to_grid(X, Z)

            if 0 <= gx < self.cols and 0 <= gy < self.rows:

                # Inflate obstacle for robot safety
                for dx in range(-inflation_cells, inflation_cells + 1):
                    for dy in range(-inflation_cells, inflation_cells + 1):

                        nx = gx + dx
                        ny = gy + dy

                        if 0 <= nx < self.cols and 0 <= ny < self.rows:
                            self.grid[ny][nx] = 1

        return self.grid

    # ----------------------------------------------------
    # Optional: visualize grid as image
    # ----------------------------------------------------
    def get_grid_image(self):
        """
        Returns grid as 0-255 image for OpenCV display
        """
        return (self.grid * 255).astype(np.uint8)