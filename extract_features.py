import cv2
import pandas as pd
import numpy as np
import os

def extract_body_metrics(image_path):
    # 1. Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0, 0
    
    # 2. Blur to remove background noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 3. Thresholding (From-scratch method to separate body from background)
    # Otsu's method automatically calculates the best contrast threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Find the largest contour (assuming the person is the largest object in the photo)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 5. Extract our mathematical parameters
    pixel_area = cv2.contourArea(largest_contour)
    x, y, w, h = cv2.boundingRect(largest_contour)
    pixel_height = h
    
    return pixel_area, pixel_height

# --- Main Logic ---
df = pd.read_csv('clean_dataset.csv')
features = []

print("Processing images... This might take a few minutes.")

for index, row in df.iterrows():
    # Adjust the 'Index' column name depending on what the Kaggle CSV uses for filenames
    # Example: if the image is '1.jpg', it looks for the value in the 'Index' column
    img_path = os.path.join('images', str(row['Filename'])) 
    
    if os.path.exists(img_path):
        area, p_height = extract_body_metrics(img_path)
        
        if area > 0:
            features.append({
                'pixel_area': area,
                'pixel_height': p_height,
                'real_height_cm': row['height_cm'],
                'actual_weight_kg': row['weight_kg']
            })

training_data = pd.DataFrame(features)
training_data.to_csv('final_training_features.csv', index=False)
print(f"✅ Extracted features for {len(features)} images. Saved to final_training_features.csv")