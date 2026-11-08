"""
Core Image Processing Analysis (IPA) Engine for Ancient Document Restoration.
"""

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.raw_image = None
        self.gray_image = None
        self.enhanced_image = None
        self.restored_text = None
        self.crack_mask = None
        self.metrics = {}

    def load_image(self) -> bool:
        """Loads the manuscript image from disk."""
        try:
            self.raw_image = cv2.imread(self.image_path)
            if self.raw_image is None:
                logger.error(f"Failed to load image at: {self.image_path}")
                return False
            self.gray_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
            logger.info("Image successfully loaded and converted to grayscale.")
            return True
        except Exception as e:
            logger.exception(f"Exception occurred while loading image: {e}")
            return False

    def enhance_contrast(self, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> None:
        """Applies CLAHE (Contrast-Limited Adaptive Histogram Equalization) to correct uneven lighting."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        self.enhanced_image = clahe.apply(self.gray_image)
        logger.info("CLAHE contrast enhancement applied successfully.")

    def restore_text(self, window_size: int = 25) -> None:
        """Extracts faded text using Sauvola's local adaptive thresholding."""
        if self.enhanced_image is None:
            raise ValueError("Image must be enhanced before text restoration.")
        
        thresh_sauvola = threshold_sauvola(self.enhanced_image, window_size=window_size)
        # Convert boolean mask to 8-bit binary image (255 for background, 0 for text)
        self.restored_text = (self.enhanced_image > thresh_sauvola).astype(np.uint8) * 255
        logger.info("Text binarization and restoration completed via Sauvola thresholding.")

    def detect_surface_degradation(self, kernel_size: int = 15, threshold_val: int = 35) -> None:
        """Isolates micro-cracks, fading, and structural damage using morphological Black-Hat transforms."""
        if self.enhanced_image is None:
            raise ValueError("Image must be enhanced before degradation mapping.")
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(self.enhanced_image, cv2.MORPH_BLACKHAT, kernel)
        
        _, self.crack_mask = cv2.threshold(blackhat, threshold_val, 255, cv2.THRESH_BINARY)
        logger.info("Surface damage and crack mask extracted using Black-Hat transforms.")

    def compute_metrics(self) -> dict:
        """Calculates quantitative metrics for degradation percentage and text density."""
        total_pixels = self.gray_image.size
        text_pixels = np.sum(self.restored_text == 0) # Dark pixels indicate text
        crack_pixels = np.sum(self.crack_mask > 0)

        self.metrics = {
            "total_pixels": int(total_pixels),
            "text_pixel_count": int(text_pixels),
            "crack_pixel_count": int(crack_pixels),
            "text_density_percentage": round((text_pixels / total_pixels) * 100, 2),
            "damage_density_percentage": round((crack_pixels / total_pixels) * 100, 2)
        }
        logger.info("Quantitative metrics successfully computed.")
        return self.metrics

    def process_pipeline(self) -> dict:
        """Executes the complete end-to-end IPA pipeline."""
        if not self.load_image():
            return {}
        self.enhance_contrast()
        self.restore_text()
        self.detect_surface_degradation()
        return self.compute_metrics()