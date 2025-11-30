import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
from ultralytics import YOLO
import threading
import time

# --- Configuration ---
# 🚨 IMPORTANT: Replace with the actual path to your trained model!
MODEL_PATH = r"C:\Users\bensh\Desktop\projects\sohibe\YOLOv8_Custom_Project1\yolov8n_feature_run_v3_100epochs\weights\best.pt" 
CAMERA_INDEX = 0 
CONFIDENCE_THRESHOLD = 0.8
# Set a low delay (in ms) to poll for new frames (e.g., 1000/30 FPS)
UPDATE_DELAY_MS = 33 
# ---------------------

class YOLOv8_App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # 1. Initialize YOLOv8 Model and Video Source
        self.model = YOLO(MODEL_PATH)
        self.class_names = self.model.names
        self.vid = cv2.VideoCapture(CAMERA_INDEX)
        
        if not self.vid.isOpened():
            raise ValueError("Could not open video source (webcam).")

        # Get video source properties
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # --- GUI Layout Setup ---
        main_frame = ttk.Frame(window, padding="10")
        main_frame.pack()

        # Left side: Video Feed
        self.canvas = tk.Canvas(main_frame, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Right side: Counter Display
        counter_frame = ttk.Frame(main_frame, padding="10")
        counter_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        ttk.Label(counter_frame, text="Defect Counts (Per Frame)", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Initialize Tkinter variables for the counter display
        self.count_vars = {name: tk.StringVar(value=f"{name.capitalize()}: 0") for name in self.class_names.values()}
        
        for name in self.class_names.values():
            ttk.Label(counter_frame, textvariable=self.count_vars[name], font=("Arial", 12)).pack(anchor="w", pady=5)
            
        # Start the video update loop using Tkinter's root.after()
        self.delay = UPDATE_DELAY_MS 
        self.update_frame()

        # Bind closing event to release the webcam
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    # --- Video Processing Function ---
    def update_frame(self):
        """Called repeatedly by Tkinter to read frame, run inference, and update GUI."""
        ret, frame = self.vid.read()

        if ret:
            # 1. Run YOLOv8 Inference
            results = self.model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            # 2. Update Counts and Annotate Frame
            current_counts = {}
            annotated_frame = frame.copy() # Use a copy for drawing if detections exist

            if results and len(results[0].boxes):
                res = results[0]
                
                # Update frame with bounding boxes using YOLO's plot method
                annotated_frame = res.plot() 
                
                # Populate the dictionary with counts
                for box in res.boxes:
                    class_label = self.class_names[int(box.cls[0])]
                    current_counts[class_label] = current_counts.get(class_label, 0) + 1
            
            # 3. Convert image for Tkinter
            cv_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = PIL.Image.fromarray(cv_image)
            self.photo = PIL.ImageTk.PhotoImage(image=pil_image)
            
            # 4. Update Canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
            # 5. Update Tkinter Counter Labels
            for name, count_var in self.count_vars.items():
                count = current_counts.get(name, 0)
                count_var.set(f"{name.capitalize()}: {count}")

        # Schedule the next frame update
        self.window.after(self.delay, self.update_frame)

    def on_closing(self):
        """Handles the window closing event to safely release the camera."""
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    # The application starts here:
    app = YOLOv8_App(root, window_title="YOLOv8 Defect Detection Counter (Tkinter)")