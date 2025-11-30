import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🤖",
    layout="centered"
)

# --- 2. LOAD MODEL ---
# Using @st.cache_resource to load the model only once and improve performance
@st.cache_resource
def load_model(model_path):
    """
    Loads the YOLO model. 
    Replace 'yolov8n.pt' with the path to your trained model (e.g., 'runs/detect/train/weights/best.pt')
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 3. UI LAYOUT ---
st.title("🤖 YOLO Object Detection App")
st.markdown("""
    Upload an image, and the model will detect objects and draw bounding boxes around them.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Option to swap model path easily
    # Default is 'yolov8n.pt' (standard small model)
    # CHANGE THIS to your custom model path, e.g., 'my_custom_model.pt'
    model_path = st.text_input("Model Path", value=r"C:\Users\bensh\Desktop\projects\sohibe\mewYOLOv8_Stitch_Defects_Optimized\v11_dropout_aug_fix2\weights\best.pt") 
    
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    st.info("Ensure your model file (.pt) is in the same directory or provide the full path.")

# --- 4. MAIN LOGIC ---

# Load the model
model = load_model(model_path)

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and model is not None:
    # Display the original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        # Open the image using PIL
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Perform Prediction
    if st.button("Detect Objects", type="primary"):
        with st.spinner("Processing..."):
            try:
                # Run inference
                # stream=False ensures we get the full result object to manipulate
                # Try increasing imgsz if you trained on larger images or have small objects
                results = model.predict(image, conf=conf_threshold, imgsz=1280)

                # Visualize the results
                # plot() returns a numpy array of the image with drawn boxes
                # BGR to RGB conversion might be needed depending on opencv version, 
                # but usually plot() returns BGR for cv2 compatibility.
                res_plotted = results[0].plot()
                
                # Convert BGR (OpenCV format) to RGB (Streamlit format)
                res_plotted_rgb = res_plotted[:, :, ::-1]

                # Display Result
                with col2:
                    st.subheader("Detection Result")
                    st.image(res_plotted_rgb, caption="Detected Objects", use_container_width=True)
                
                # Optional: Show detection data
                with st.expander("See Detection Details"):
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Get class name
                            cls = int(box.cls[0])
                            cls_name = model.names[cls]
                            conf = float(box.conf[0])
                            st.write(f"Detected **{cls_name}** with **{conf:.2f}** confidence")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

else:
    if model is None:
        st.warning("Please check your model path in the sidebar.")
    else:
        st.info("Please upload an image to begin.")