import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

# Load YOLOv8 model trained for pose detection
model = YOLO("yolov8m-pose.pt")

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Set color resolution and FPS
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Enable depth stream
pipeline.start(config)

# Align depth to color for accurate depth estimation on color keypoints
align = rs.align(rs.stream.color)

try:
    # Loop to process video stream
    while True:
        # Get frames from RealSense camera
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # Align depth to color

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

        # Get depth frame dimensions
        depth_width = depth_frame.get_width()
        depth_height = depth_frame.get_height()

        # Process each detected person
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints
                for person_keypoints in keypoints.data:
                    # Get keypoints for left/right shoulders and nose
                    left_shoulder = person_keypoints[5]
                    right_shoulder = person_keypoints[6]
                    nose = person_keypoints[0]

                    # Check confidence and validity
                    if (
                        left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5 and
                        nose[2] > 0.5
                    ):
                        x1, y1 = int(left_shoulder[0]), int(left_shoulder[1])
                        x2, y2 = int(right_shoulder[0]), int(right_shoulder[1])
                        x_nose, y_nose = int(nose[0]), int(nose[1])

                        # Validate coordinates before retrieving depth
                        if (
                            0 <= x1 < depth_width and 0 <= y1 < depth_height and
                            0 <= x2 < depth_width and 0 <= y2 < depth_height and
                            0 <= x_nose < depth_width and 0 <= y_nose < depth_height
                        ):
                            z1 = depth_frame.get_distance(x1, y1)
                            z2 = depth_frame.get_distance(x2, y2)
                            z_nose = depth_frame.get_distance(x_nose, y_nose)
                        else:
                            z1 = z2 = z_nose = 0  # Default values if coordinates are out of bounds

                        # Calculate center points
                        center_x_12, center_y_12 = (x1 + x2) // 2, (y1 + y2) // 2

                        # Calculate vector directions and cross-product
                        vector1 = np.array([x2 - x1, y2 - y1, z2 - z1])
                        vector2 = np.array([
                            x_nose - center_x_12,
                            y_nose - center_y_12,
                            z_nose - ((z1 + z2) / 2)
                        ])
                        cross_product = np.cross(vector1, vector2)

                        # Normalize cross-product for display
                        if np.linalg.norm(cross_product[:2]) > 0:
                            normalized_vector = -cross_product[:2] / np.linalg.norm(cross_product[:2]) * 50
                        else:
                            normalized_vector = [0, 0]

                        # Calculate angle
                        angle = np.degrees(np.arctan2(vector2[1], vector2[0]))

                        # Draw keypoints and vector information
                        cv2.circle(image, (center_x_12, center_y_12), 5, (0, 0, 255), -1)  # Red center
                        endpoint = (
                            int(center_x_12 + normalized_vector[0]),
                            int(center_y_12 + normalized_vector[1])
                        )
                        cv2.arrowedLine(image, (center_x_12, center_y_12), endpoint, (0, 255, 0), 2, tipLength=0.3)

                        # Display angle, position, and depth
                        cv2.putText(image, f"Angle: {angle:.2f} degrees", (x_nose, y_nose - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(image, f"Depth (Nose): {z_nose:.2f} meters", (x_nose, y_nose + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Draw all keypoints
                    for kp in person_keypoints:
                        x, y, conf = kp.tolist()
                        if conf > 0.5 and 0 <= x < depth_width and 0 <= y < depth_height:
                            cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)  # Blue for keypoints

        # Display the frame with overlaid information
        cv2.imshow("RealSense D435 Stream", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the camera
    pipeline.stop()
    cv2.destroyAllWindows()
