import pyrealsense2 as rs
import numpy as np

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):

        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # 🔹 Start pipeline
        self.profile = self.pipeline.start(config)

        # 🔹 Get intrinsics AFTER starting stream
        color_stream = self.profile.get_stream(rs.stream.color)
        color_profile = color_stream.as_video_stream_profile()
        self.intrinsics = color_profile.get_intrinsics()

        # 🔹 Align depth to color
        self.align = rs.align(rs.stream.color)

        print("[INFO] RealSense camera started")
        print(f"[INFO] fx: {self.intrinsics.fx}, fy: {self.intrinsics.fy}")
        print(f"[INFO] cx: {self.intrinsics.ppx}, cy: {self.intrinsics.ppy}")

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_frame

    def stop(self):
        self.pipeline.stop()
        print("[INFO] RealSense stopped")