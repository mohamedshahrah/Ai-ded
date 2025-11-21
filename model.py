import os
from ultralytics import YOLO

# --- Configuration Section ---
# 1. Model and Data
MODEL_WEIGHTS = 'yolov8n.pt'          
# Use the correct, fixed absolute path for data.yaml (from your error log)
DATA_CONFIG = r"C:\Users\bensh\Desktop\sohibe\supply\Detection des tachess.v1i.yolov5pytorch\data.yaml"

# 2. Training Parameters
TOTAL_EPOCHS = 50                  
IMAGE_SIZE = 640                      
BATCH_SIZE = 8                      
DATALOADER_WORKERS = 4 

# NOTE: Class weights cannot be reliably injected in your current version.
# We will focus on improving box precision and training length instead.

# 3. Custom Hyperparameters (Passed Directly)
BOX_LOSS_GAIN = 10.0      # Increased from 7.5 to improve box precision
DFL_LOSS_GAIN = 3.0       # Increased from 1.5 to improve box precision
LRF_FINAL = 0.000001      # Very low final LR for precise fine-tuning

# 4. Environment & Logging
DEVICE = 0                            
PROJECT_NAME = 'mewYOLOv8_Stitch_Defects' 
EXPERIMENT_NAME = 'v9_direct_args_final1' # Final attempt name
RESUME_FLAG = False                   
# -----------------------------


def train_yolov8_optimized(
    model_path: str, 
    data_path: str, 
    epochs: int, 
    batch: int, 
    workers: int
):
    """Loads YOLOv8 and starts training, injecting hyperparameters directly."""
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Using data path: {data_path}")
    print(f"Set DATALOADER_WORKERS: {workers}")
    print(f"Set box={BOX_LOSS_GAIN}, dfl={DFL_LOSS_GAIN}, lrf={LRF_FINAL}")
    print("\nStarting FINAL Optimized Training...")
    
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        
        # --- Direct Hyperparameter Overrides (Focusing on non-rejected arguments) ---
        box=BOX_LOSS_GAIN,
        dfl=DFL_LOSS_GAIN,
        lrf=LRF_FINAL, 
        
        # --- Datasets and Resource Control ---
        workers=workers,           
        device=DEVICE,
        
        # --- Logging and Checkpointing ---
        save=True,          
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        resume=RESUME_FLAG,
        plots=True,            
        val=True,              
        
        # --- Augmentation ---
        mosaic=1.0,
        close_mosaic=10, 
        fliplr=0.5,
    )
    
    print("\n✅ Training run finished. Your best model is saved in the 'weights' folder.")
    print(f"   Look in: runs/detect/{EXPERIMENT_NAME}")
    return results

if __name__ == '__main__':
    try:
        training_results = train_yolov8_optimized(
            model_path=MODEL_WEIGHTS,
            data_path=DATA_CONFIG,
            epochs=TOTAL_EPOCHS,
            batch=BATCH_SIZE,
            workers=DATALOADER_WORKERS
        )
    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}")
        print("This is the final known configuration. If it fails, please consider updating your Ultralytics package.")