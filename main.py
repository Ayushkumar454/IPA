"""
Main Execution Script for Marine Archaeological Image Processing Analysis (Dwarka Project)
"""

import os
import cv2
import logging
from src.processor3 import UnderwaterDwarkaProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Automatically locate the input_data folder inside src/
    input_dir = os.path.join("src", "input_data")
    
    if not os.path.exists(input_dir):
        logger.error(f"Directory not found: {input_dir}")
        return

    # Get all valid image files in the directory
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        logger.error(f"No valid images found in {input_dir}")
        return

    # Automatically pick the first available image
    image_filename = image_files[0]
    image_path = os.path.join(input_dir, image_filename)

    logger.info(f"Automatically selected image for processing: {image_filename}")

    # Instantiate the processor
    processor = UnderwaterDwarkaProcessor(image_path)

    # Run the processing pipeline
    metrics = processor.process_pipeline()

    if metrics:
        logger.info("Pipeline executed successfully!")
        print("\n--- Quantitative Archaeological Metrics ---")
        for key, value in metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        # Save output images to an outputs/ folder
        os.makedirs("outputs", exist_ok=True)
        
        if hasattr(processor, 'edge_map') and processor.edge_map is not None:
            edge_output_path = os.path.join("outputs", f"edge_map_{image_filename}.jpg")
            cv2.imwrite(edge_output_path, processor.edge_map)
            logger.info(f"Saved edge map to {edge_output_path}")

        if hasattr(processor, 'structure_mask') and processor.structure_mask is not None:
            mask_output_path = os.path.join("outputs", f"mask_{image_filename}.jpg")
            cv2.imwrite(mask_output_path, processor.structure_mask)
            logger.info(f"Saved segmentation mask to {mask_output_path}")
            
    else:
        logger.error("Pipeline execution failed.")

if __name__ == "__main__":
    main()