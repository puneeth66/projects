import cv2
import numpy as np

def darken_bright_spots(frame, bright_spots):
    mask = np.zeros_like(frame, dtype=np.uint8)  # Create a blank mask

    for (x, y, w, h) in bright_spots:
        center = (x + w // 2, y + h // 2)  # Find center of bright spot
        radius = max(w, h) // 2  # Adjust circle size based on bright region

        # Draw a filled dark circle on the mask
        cv2.circle(mask, center, radius, (50, 50, 50), -1)  # Dark gray circle

    # Blend the mask with the original frame to reduce brightness
    filtered_frame = cv2.addWeighted(frame, 1.0, mask, -0.7, 0)  

    return filtered_frame, mask  # Return both the filtered frame and mask
