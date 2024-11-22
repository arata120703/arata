import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import time  # Import thêm thư viện time

# Load YOLOv8 model trained for pose detection
model = YOLO("yolo11m-pose.pt")

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Set color resolution and FPS
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Enable depth stream
pipeline.start(config)

# Initialize arrays for tracking
timestamps = []  # List lưu mốc thời gian
y_points = np.array([])  # Mảng lưu tọa độ Y của mũi

# Setup matplotlib for interactive plotting
plt.ion()
fig, ax = plt.subplots()
trajectory_line, = ax.plot([], [], 'bo-', markersize=3)  # Line object

# Align depth to color for accurate depth estimation on color keypoints
align = rs.align(rs.stream.color)

try:
    while True:
        # Get frames from RealSense camera
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Extract color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert color frame to numpy array
        frame = np.asanyarray(color_frame.get_data())
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect keypoints with YOLOv8
        results = model(image)

        # Process each detected person
        for result in results:
            if result.keypoints is not None and len(result.keypoints.data) > 0:  # Check keypoints exist
                keypoints = result.keypoints
                for person_keypoints in keypoints.data:
                    # Get nose keypoint
                    nose = person_keypoints[0]  # Keypoint 0 is nose
                    if nose[2] > 0.5:  # Confidence > 0.5
                        x_nose, y_nose = int(nose[0]), int(nose[1])

                        # Append data
                        timestamps.append(time.time())  # Lưu thời gian hiện tại
                        y_points = np.append(y_points, y_nose)  # Lưu tọa độ Y

                        # Update plot
                        trajectory_line.set_data(timestamps, y_points)
                        ax.relim()
                        ax.autoscale_view()
                        ax.set_xlabel("Time (seconds)")  # Đặt nhãn cho trục X
                        ax.set_ylabel("Y Coordinate")  # Đặt nhãn cho trục Y
                        plt.pause(0.01)

                        # Draw nose keypoint on frame
                        cv2.circle(image, (x_nose, y_nose), 5, (255, 0, 0), -1)  # Draw nose

        # Display the frame with overlaid information
        cv2.imshow("RealSense D435 Stream", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the camera
    pipeline.stop()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()
