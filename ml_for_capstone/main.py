import cv2
from detect_bright_spots import detect_bright_spots
from apply_filter import darken_bright_spots

cap = cv2.VideoCapture(0)  # Open camera (change to video path if needed)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    bright_spots = detect_bright_spots(frame)  # Detect bright areas
    filtered_frame, mask = darken_bright_spots(frame, bright_spots)  # Apply dark circles

    # Combine original and filtered frames side by side
    combined_frame = cv2.hconcat([frame, filtered_frame])

    # Convert mask to grayscale for proper display
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Show all windows
    cv2.imshow("Before & After Filtering", combined_frame)  # Original vs Filtered
    cv2.imshow("Dark Circles Only", gray_mask)  # Only Dark Circles

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
