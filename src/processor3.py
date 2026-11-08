"""
Advanced Marine Image Processing Engine for Submerged Archaeological Sites (Dwarka Project).
"""

import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

class UnderwaterDwarkaProcessor:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.raw_image = None
        self.corrected_image = None
        self.gray_image = None
        self.edge_map = None
        self.structure_mask = None
        self.metrics = {}

    def load_image(self) -> bool:
        """Loads the underwater Dwarka site image."""
        try:
            self.raw_image = cv2.imread(self.image_path)
            if self.raw_image is None:
                logger.error(f"Failed to load image at: {self.image_path}")
                return False
            logger.info("Underwater site image successfully loaded.")
            return True
        except Exception as e:
            logger.exception(f"Exception occurred while loading image: {e}")
            return False

    def corrective_color_balance(self) -> None:
        """Corrects the blue-green underwater color cast using LAB color space channel stretching."""
        # Convert to LAB color space to handle luminance (L) and color channels (A, B) separately
        lab = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE specifically to the luminance channel to reveal details under murky water
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_cloned = clahe.apply(l_channel)

        # Merge back and convert to grayscale for structural analysis
        merged_lab = cv2.merge((l_cloned, a_channel, b_channel))
        self.corrected_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        self.gray_image = cv2.cvtColor(self.corrected_image, cv2.COLOR_BGR2GRAY)
        logger.info("Underwater color cast corrected and luminance enhanced.")

    def detect_submerged_structures(self) -> None:
        """Applies Canny Edge Detection optimized for stone blocks and anchor outlines on the seabed."""
        # Bilateral filter clears out marine sediment/plankton noise while keeping stone block edges sharp
        smoothed = cv2.bilateralFilter(self.gray_image, d=9, sigmaColor=75, sigmaSpace=75)
        self.edge_map = cv2.Canny(smoothed, threshold1=40, threshold2=120)
        logger.info("Submerged architectural edges and stone blocks mapped.")

    def segment_seafloor_objects(self) -> None:
        """Segments structural stone artifacts from the background seabed using Otsu thresholding."""
        _, self.structure_mask = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info("Seafloor objects and structures segmented.")

    def compute_metrics(self) -> dict:
        """Calculates quantitative metrics for the submerged site (e.g., exposed stone surface density)."""
        total_pixels = self.gray_image.size
        edge_pixels = np.sum(self.edge_map > 0)
        structure_pixels = np.sum(self.structure_mask == 255)

        self.metrics = {
            "total_pixels": int(total_pixels),
            "architectural_edge_complexity": int(edge_pixels),
            "exposed_structure_density": round((structure_pixels / total_pixels) * 100, 2),
            "structural_sharpness_ratio": round((edge_pixels / total_pixels) * 100, 2)
        }
        logger.info("Marine archaeological metrics successfully computed.")
        return self.metrics

    def process_pipeline(self) -> dict:
        if not self.load_image():
            return {}
        self.corrective_color_balance()
        self.detect_submerged_structures()
        self.segment_seafloor_objects()
        return self.compute_metrics()