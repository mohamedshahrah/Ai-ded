import os
from ultralytics import YOLO

# --- Configuration Section ---
# 1. Model and Data
MODEL_WEIGHTS = 'yolov8n.pt'          
# Use the correct, fixed absolute path for data.yaml
DATA_CONFIG = r"C:\Users\bensh\Desktop\projects\sohibe\Stitch Vision.v2-roboflow-instant-1--eval-.yolov8-obb\data.yaml"

# 2. Training Parameters
TOTAL_EPOCHS = 100                  # User requested 10 epochs
IMAGE_SIZE = 640                   # Standard size
BATCH_SIZE = 8                     # Reduced to 4 to fix Memory Error
DATALOADER_WORKERS = 1             # Reduced to 1 to fix Memory Error on Windows

# 3. Custom Hyperparameters (Optimized for Accuracy)
BOX_LOSS_GAIN = 7.5                # Standard gain
DFL_LOSS_GAIN = 1.5                # Standard gain
LR0 = 0.001                        # Initial learning rate
OPTIMIZER = 'AdamW'                # AdamW is often better for convergence

# 4. Environment & Logging
DEVICE = 0                            
PROJECT_NAME = 'mewYOLOv8_Stitch_Defects_Optimized' 
EXPERIMENT_NAME = 'v11_dropout_aug_fix' 
RESUME_FLAG = False                   

def train_yolov8_optimized(
    model_path: str, 
    data_path: str, 
    epochs: int, 
    batch: int, 
    workers: int
):
    """Loads YOLOv8 and starts training with optimized hyperparameters."""
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Using data path: {data_path}")
    print(f"Set DATALOADER_WORKERS: {workers}")
    print(f"Set BATCH_SIZE: {batch}")
    print("\nStarting FINAL Optimized Training...")
    
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=IMAGE_SIZE,
        batch=batch,
        
        # --- Memory & System Stability ---
        workers=workers,           
        device=DEVICE,
        cache=False,          # Disable RAM caching to prevent OOM
        
        # --- Optimization ---
        optimizer=OPTIMIZER,
        lr0=LR0,
        patience=5,           # Early stopping
        
        # --- Logging ---
        save=True,          
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        resume=RESUME_FLAG,
        plots=True,            
        val=True,              
        
        # --- Regularization (Prevent Overfitting) ---
        dropout=0.15,         # 15% Dropout
        
        # --- Augmentations (Improve Generalization) ---
        mosaic=1.0,           # Strong data augmentation
        mixup=0.1,            # Mix images
        degrees=15.0,         # Rotate +/- 15 degrees
        translate=0.1,        # Translate +/- 10%
        scale=0.5,            # Scale +/- 50%
        fliplr=0.5,           # Flip left-right (enabled if appropriate for defects)
        flipud=0.0,           # Flip up-down
        copy_paste=0.1,       # Copy-Paste augmentation
        
        # --- Loss Gains ---
        box=BOX_LOSS_GAIN,
        dfl=DFL_LOSS_GAIN,
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
        print("Try reducing BATCH_SIZE further if this persists.")