import cv2
import pandas as pd
import numpy as np
import joblib
from ultralytics import YOLO
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# --- Load Models ---
# Change this to 'weight_predictor.pkl' if that is your latest file name
MODEL_NAME = 'weight_predictor_internal.pkl'

if not os.path.exists(MODEL_NAME):
    print(f"❌ Error: {MODEL_NAME} not found. Please check your folder.")
else:
    print(f"✅ Loading {MODEL_NAME}...")
    model_ai = joblib.load(MODEL_NAME)
    yolo_pose = YOLO('yolov8n-pose.pt')

def calculate_distance(pt1, pt2):
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

class WeightApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Weight Predictor - Desktop Tester")
        self.root.geometry("900x800")

        # 1. Prediction Display (Moved to Top so it's always visible)
        self.result_frame = tk.Frame(root, bg="#f0f0f0", bd=2, relief="groove")
        self.result_frame.pack(fill="x", padx=10, pady=10)
        
        self.result_text = tk.Label(self.result_frame, text="Predicted Weight: -- kg", 
                                   font=("Arial", 20, "bold"), fg="#2c3e50", bg="#f0f0f0")
        self.result_text.pack(pady=10)

        # 2. Input Fields
        input_frame = tk.Frame(root)
        input_frame.pack(pady=10)

        tk.Label(input_frame, text="Height (cm):", font=("Arial", 10)).grid(row=0, column=0, padx=5)
        self.entry_height = tk.Entry(input_frame)
        self.entry_height.insert(0, "172") # Default value
        self.entry_height.grid(row=0, column=1)

        tk.Label(input_frame, text="Age:", font=("Arial", 10)).grid(row=1, column=0, padx=5)
        self.entry_age = tk.Entry(input_frame)
        self.entry_age.insert(0, "23") # Default value
        self.entry_age.grid(row=1, column=1)

        tk.Label(input_frame, text="Gender (0:M, 1:F):", font=("Arial", 10)).grid(row=2, column=0, padx=5)
        self.entry_gender = tk.Entry(input_frame)
        self.entry_gender.insert(0, "0") # Default value
        self.entry_gender.grid(row=2, column=1)

        # 3. Action Button
        self.btn_upload = tk.Button(root, text="SELECT PHOTO & PREDICT", font=("Arial", 12, "bold"),
                                   command=self.process_image, bg="#27ae60", fg="white", height=2, width=25)
        self.btn_upload.pack(pady=10)

        # 4. Canvas for Image
        self.canvas = tk.Canvas(root, width=500, height=500, bg="#bdc3c7")
        self.canvas.pack(expand=True)

    def process_image(self):
        try:
            h = float(self.entry_height.get())
            a = int(self.entry_age.get())
            g = int(self.entry_gender.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for Height, Age, and Gender.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path: return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Could not read the image file.")
            return
            
        display_img = img.copy()
        results = yolo_pose(img, verbose=False)
        
        if not results or results[0].keypoints is None or len(results[0].keypoints.xy[0]) < 17:
            messagebox.showwarning("Warning", "YOLO could not detect a person or skeleton.")
            return

        # Extract Joints
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        l_sh, r_sh = kpts[5], kpts[6]
        l_hp, r_hp = kpts[11], kpts[12]

        # Draw Skeleton Circles
        for pt in [l_sh, r_sh, l_hp, r_hp]:
            cv2.circle(display_img, (int(pt[0]), int(pt[1])), 12, (0, 255, 0), -1)
        # Draw Shoulder Line
        cv2.line(display_img, (int(l_sh[0]), int(l_sh[1])), (int(r_sh[0]), int(r_sh[1])), (255, 0, 0), 5)

        # Logic & Math
        sh_width = calculate_distance(l_sh, r_sh)
        hp_width = calculate_distance(l_hp, r_hp)
        mid_sh = [(l_sh[0] + r_sh[0])/2, (l_sh[1] + r_sh[1])/2]
        mid_hp = [(l_hp[0] + r_hp[0])/2, (l_hp[1] + r_hp[1])/2]
        torso = calculate_distance(mid_sh, mid_hp)
        
        if torso == 0 or hp_width == 0:
            messagebox.showwarning("Warning", "Invalid skeleton detection (torso or hip width is zero).")
            return

        # Ratios
        sh_t_ratio = sh_width / torso
        hp_t_ratio = hp_width / torso
        sh_hp_ratio = sh_width / hp_width

        # 5. Prediction
        # Columns must match EXACTLY what was used during model training
        input_cols = ['height', 'gender', 'age', 'shoulder_torso_ratio', 'hip_torso_ratio', 'shoulder_hip_ratio']
        input_data = pd.DataFrame([[h, g, a, sh_t_ratio, hp_t_ratio, sh_hp_ratio]], columns=input_cols)
        
        prediction = model_ai.predict(input_data)[0]
        
        # Update UI & Console
        print(f"--- Prediction for {os.path.basename(file_path)} ---")
        print(f"Weight: {prediction:.2f} kg")
        self.result_text.config(text=f"Predicted Weight: {prediction:.2f} kg", fg="#e67e22")
        
        # Resize image to fit nicely in the Canvas
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = display_img.shape[:2]
        # Resize to a max height of 500 while maintaining aspect ratio
        scale = 500 / h_orig
        new_w, new_h = int(w_orig * scale), 500
        display_img = cv2.resize(display_img, (new_w, new_h))
        
        img_tk = ImageTk.PhotoImage(Image.fromarray(display_img))
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk
        
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = WeightApp(root)
    root.mainloop()