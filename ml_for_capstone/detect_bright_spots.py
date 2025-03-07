import cv2

def detect_bright_spots(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bright_spots = [(cv2.boundingRect(cnt)) for cnt in contours if cv2.contourArea(cnt) > 50]
    
    return bright_spots
