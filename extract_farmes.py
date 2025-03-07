import cv2
import os

video_folder = r"C:\Users\Puneeth\Desktop\hack\video"
output_folder = r"C:\Users\Puneeth\Desktop\hack\dataset"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# List all video files in the folder
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

frame_count = 0

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open {video_file}")
        continue

    print(f"Processing {video_file}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

print(f"Frames extracted and saved in {output_folder}")
