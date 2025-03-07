import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from glob import glob
import os

# Initialize ResNet50
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Directory with extracted frames
image_paths = glob("C:/Users/Puneeth/Desktop/hack/dataset/*.jpg")
features = []

# Extract features using ResNet50
for idx, path in enumerate(image_paths):
    img = image.load_img(path, target_size=(224, 224))  # Resize images for ResNet50
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get features using ResNet50 model
    feature = model.predict(img_array)
    features.append(feature.flatten())  # Flatten the feature for clustering

    # Print progress
    print(f"Processed {idx + 1}/{len(image_paths)} frames.", end='\r')  # Overwrite the same line

features = np.array(features)

# Perform K-Means Clustering to classify frames into 2 categories
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(features)

# Create directories for bright spots and normal frames
os.makedirs("C:/Users/Puneeth/Desktop/hack/dataset/bright_spots", exist_ok=True)
os.makedirs("C:/Users/Puneeth/Desktop/hack/dataset/normal", exist_ok=True)

# Initialize counters for classified frames
bright_spots_count = 0
normal_count = 0

# Move frames based on cluster label and count them
for i, path in enumerate(image_paths):
    folder = "bright_spots" if labels[i] == 1 else "normal"
    os.rename(path, f"C:/Users/Puneeth/Desktop/hack/dataset/{folder}/{os.path.basename(path)}")
    
    # Increment the respective counter
    if folder == "bright_spots":
        bright_spots_count += 1
    else:
        normal_count += 1

# Print the count of classified frames
print(f"\nFrames classified into 'bright_spots': {bright_spots_count}")
print(f"Frames classified into 'normal': {normal_count}")
