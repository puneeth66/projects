import cv2
import numpy as np
import os
from skimage.feature import hog
from skimage.color import rgb2gray

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

def load_data(data_folder, image_size=(128, 128)):
    """Load images and labels from the dataset folders."""
    images = []
    labels = []
    
    total_images = sum(len(files) for _, _, files in os.walk(data_folder))  # Total images count
    processed_images = 0

    for label in ["normal", "bright_spots"]:
        folder_path = os.path.join(data_folder, label)
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    # Resize image
                    image = cv2.resize(image, image_size)

                    # Extract features from the image
                    color_histogram = extract_color_histogram(image)
                    hog_features = extract_hog_features(image)

                    # Combine features into a single vector
                    features = np.hstack((color_histogram, hog_features))
                    
                    images.append(features)
                    labels.append(0 if label == "normal" else 1)

                    processed_images += 1
                    progress_percentage = (processed_images / total_images) * 100
                    print(f"Processed {processed_images}/{total_images} images ({progress_percentage:.2f}%).")
                else:
                    print(f"Warning: Unable to read image {image_path}. Skipping.")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    return np.array(images), np.array(labels)

# Load data
data_folder = r"C:\Users\Puneeth\Desktop\hack\dataset"  # Path to your dataset
X, y = load_data(data_folder)

print(f"Loaded {len(X)} images with labels.")
print(f"Feature vector shape: {X.shape[1]}, Labels shape: {y.shape}")
