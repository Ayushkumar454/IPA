import streamlit as st
import os
from PIL import Image
from src.processor3 import UnderwaterDwarkaProcessor

st.set_page_config(page_title="Marine Archaeology IPA Dashboard", layout="wide")

st.title("🌊 Underwater Image Processing Analysis (IPA)")
st.markdown("### Archaeological Feature Extraction & Metric Visualizer")

# Sidebar for navigation
st.sidebar.header("Pipeline Controls")
input_dir = os.path.join("src", "input_data")
output_dir = "outputs"

if os.path.exists(input_dir):
    images = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    selected_image = st.sidebar.selectbox("Select Survey Image", images)
    
    if selected_image:
        col1, col2 = st.columns(2)
        
        orig_path = os.path.join(input_dir, selected_image)
        with col1:
            st.subheader("Original Submerged View")
            st.image(Image.open(orig_path))
            
        with col2:
            st.subheader("Processed Edge Map / Mask")
            edge_filename = f"edge_map_{selected_image}.jpg"
            edge_path = os.path.join(output_dir, edge_filename)
            
            if os.path.exists(edge_path):
                st.image(Image.open(edge_path))
            else:
                st.info("Run `python main.py` to generate outputs for this image.")

        st.markdown("---")
        st.subheader("📊 Complete Quantitative Archaeological Metrics")
        
        # Run processor on the selected image to get all metrics live
        image_path = os.path.join(input_dir, selected_image)
        processor = UnderwaterDwarkaProcessor(image_path)
        metrics = processor.process_pipeline()
        
        if metrics:
            # Dynamically create columns for ALL metrics returned by the processor
            cols = st.columns(len(metrics))
            for i, (key, value) in enumerate(metrics.items()):
                display_name = key.replace("_", " ").title()
                cols[i].metric(display_name, str(value))
                
            # Also show a complete structured table of all details
            with st.expander("View Full Raw Metrics Data Report"):
                st.json(metrics)
        else:
            st.warning("Could not compute metrics for this image.")
else:
    st.warning("Please ensure your `src/input_data` folder is set up.")