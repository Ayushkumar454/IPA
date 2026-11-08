"""
Core Image Processing Analysis (IPA) Engine for Archaeological Excavation Sites.
"""

import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

class ArchaeologicalProcessor:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.raw_image = None
        self.gray_image = None
        self.enhanced_image = None
        self.edge_map = None
        self.segmentation_mask = None
        self.metrics = {}

    def load_image(self) -> bool:
        """Loads the excavation site image from disk."""
        try:
            self.raw_image = cv2.imread(self.image_path)
            if self.raw_image is None:
                logger.error(f"Failed to load image at: {self.image_path}")
                return False
            self.gray_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
            logger.info("Excavation site image successfully loaded.")
            return True
        except Exception as e:
            logger.exception(f"Exception occurred while loading image: {e}")
            return False

    def enhance_site_texture(self) -> None:
        """Applies bilateral filtering to smooth dirt/soil gradients while preserving sharp stone/artifact edges."""
        # Bilateral filter is brilliant for archaeology: it keeps rock edges sharp while smoothing soil texture noise.
        self.enhanced_image = cv2.bilateralFilter(self.gray_image, d=9, sigmaColor=75, sigmaSpace=75)
        logger.info("Bilateral filtering applied to preserve structural feature boundaries.")

    def detect_stratigraphy_and_boundaries(self, low_thresh: int = 50, high_thresh: int = 150) -> None:
        """Uses Canny Edge Detection to trace soil strata layers, rock boundaries, or trench walls."""
        self.edge_map = cv2.Canny(self.enhanced_image, low_thresh, high_thresh)
        logger.info("Stratigraphic boundaries and edge maps extracted.")

    def segment_features(self) -> None:
        """Applies Otsu's Thresholding to separate dark soil features/pits from lighter stones or floors."""
        _, self.segmentation_mask = cv2.threshold(self.enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info("Site feature segmentation completed via Otsu thresholding.")

    def compute_metrics(self) -> dict:
        """Calculates quantitative archaeological site metrics (e.g., exposed stone/feature ratio vs soil)."""
        total_pixels = self.gray_image.size
        edge_pixel_count = np.sum(self.edge_map > 0)
        feature_pixel_count = np.sum(self.segmentation_mask == 255)

        self.metrics = {
            "total_pixels": int(total_pixels),
            "boundary_complexity_score": int(edge_pixel_count),
            "feature_coverage_percentage": round((feature_pixel_count / total_pixels) * 100, 2),
            "structural_density_ratio": round((edge_pixel_count / total_pixels) * 100, 2)
        }
        logger.info("Archaeological quantitative metrics computed.")
        return self.metrics

    def process_pipeline(self) -> dict:
        if not self.load_image():
            return {}
        self.enhance_site_texture()
        self.detect_stratigraphy_and_boundaries()
        self.segment_features()
        return self.compute_metrics()
    