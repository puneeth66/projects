import cv2
import numpy as np
import os
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split

def extract_color_histogram(image):
    """Extract color histogram from the image."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    histogram = histogram.astype("float") / histogram.sum()  # Normalize
    return histogram

def extract_hog_features(image):
    """Extract HOG features from the image."""
    image_gray = rgb2gray(image)  # Convert to grayscale
    features = hog(image_gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
    return features

def load_data(data_folder):
    """Load images and labels from the dataset folders."""
    images = []
    labels = []
    
    for label in ["normal", "bright_spots"]:  # Use the correct folder name
        folder_path = os.path.join(data_folder, label)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                # Extract features from the image
                color_histogram = extract_color_histogram(image)
                hog_features = extract_hog_features(image)
                
                # Combine features into a single vector
                features = np.hstack((color_histogram, hog_features))
                
                images.append(features)
                labels.append(0 if label == "normal" else 1)

    return np.array(images), np.array(labels)

# Load data
data_folder = r"C:\Users\Puneeth\Desktop\hack\dataset"  # Path to your dataset
X, y = load_data(data_folder)

print(f"Loaded {len(X)} images with labels.")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate sizes of the splits
train_size = len(X_train)
test_size = len(X_test)

# Print the sizes of the splits
print(f"Training set size: {train_size} ({(train_size / len(X)) * 100:.2f}%)")
print(f"Testing set size: {test_size} ({(test_size / len(X)) * 100:.2f}%)")

# Optional: Print progress while splitting
total_images = len(X)
for i in range(total_images):
    if i % 1000 == 0:  # Print every 1000 images
        print(f"Processed {i}/{total_images} images ({(i / total_images) * 100:.2f}%).")
