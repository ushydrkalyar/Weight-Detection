from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import os
import math

print("🦴 Loading YOLOv8 AI to extract Skeletons...")
model = YOLO('yolov8n-pose.pt') 

def calculate_distance(pt1, pt2):
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

def extract_skeleton_metrics_yolo(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        results = model(img, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return None
            
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        if len(kpts) < 17: 
            return None

        # COCO Skeleton Mapping
        l_shoulder, r_shoulder = kpts[5], kpts[6]
        l_hip, r_hip = kpts[11], kpts[12]
        
        if np.all(l_shoulder == 0) or np.all(r_shoulder == 0) or np.all(l_hip == 0) or np.all(r_hip == 0):
            return None

        # Calculate Distances
        shoulder_width = calculate_distance(l_shoulder, r_shoulder)
        hip_width = calculate_distance(l_hip, r_hip)
        
        mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
        mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
        torso_length = calculate_distance(mid_shoulder, mid_hip)
        
        # Camera-Invariant Ratios
        shoulder_to_torso_ratio = shoulder_width / torso_length if torso_length > 0 else 0
        hip_to_torso_ratio = hip_width / torso_length if torso_length > 0 else 0
        shoulder_to_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 0

        return {
            'shoulder_torso_ratio': shoulder_to_torso_ratio,
            'hip_torso_ratio': hip_to_torso_ratio,
            'shoulder_hip_ratio': shoulder_to_hip_ratio
        }
    except Exception as e:
        print(f"\n⚠️ Error on {image_path}: {e}. Skipping!")
        return None

# ==========================================
# --- Main Split Processing Logic ---
# ==========================================

if __name__ == "__main__":
    # Ensure this matches your actual CSV filename
    df = pd.read_csv('metadata.csv') 
    
    train_features = []
    test_features = []

    print("🚀 Processing Custom Dataset Folders...")

    for index, row in df.iterrows():
        # Clean up strings just in case there are accidental spaces in the CSV
        split_dir = str(row['split']).strip().lower() 
        filename = str(row['filename']).strip()
        
        # This dynamically builds the path: e.g., 'train/image1.jpg' or 'test/image2.jpg'
        img_path = os.path.join(split_dir, filename)
        
        if os.path.exists(img_path):
            metrics = extract_skeleton_metrics_yolo(img_path)
            
            if metrics is not None:
                row_data = metrics.copy()
                row_data['height'] = row['height']
                row_data['weight'] = row['weight']
                row_data['gender'] = row['gender']
                row_data['age'] = row['age']
                
                # Route the data to the correct list based on the CSV split column
                if split_dir == 'train':
                    train_features.append(row_data)
                elif split_dir == 'test':
                    test_features.append(row_data)
                else:
                    print(f"⚠️ Unknown split type '{split_dir}' for {filename}")
        else:
            print(f"⚠️ Image not found: {img_path}")

        if (index + 1) % 50 == 0:
            print(f"Scanned {index + 1} images...")

    # Save to two distinct files
    pd.DataFrame(train_features).to_csv('final_train_features.csv', index=False)
    pd.DataFrame(test_features).to_csv('final_test_features.csv', index=False)
    
    print(f"\n✅ Extraction Complete!")
    print(f"🧠 Training Data: {len(train_features)} images saved to 'final_train_features.csv'")
    print(f"📝 Testing Data:  {len(test_features)} images saved to 'final_test_features.csv'")
