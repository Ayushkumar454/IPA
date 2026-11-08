"""
Visualization and Reporting Utility for IPA Project Results.
"""

import cv2
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)

class ResultVisualizer:
    @staticmethod
    def plot_and_save_results(processor, output_dir: str = "outputs"):
        """Generates a professional multi-panel figure displaying the pipeline stages and saves it."""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Cultural Heritage Image Processing Analysis (IPA) Report", fontsize=16, fontweight='bold', y=0.95)

        panels = [
            (cv2.cvtColor(processor.raw_image, cv2.COLOR_BGR2RGB), "Original Manuscript", "gray" if len(processor.raw_image.shape) == 2 else None),
            (processor.enhanced_image, "CLAHE Enhanced", "gray"),
            (processor.restored_text, "Restored Text (Sauvola)", "gray"),
            (cv2.morphologyEx(processor.enhanced_image, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))), "Black-Hat Feature Map", "gray"),
            (processor.crack_mask, f"Damage Map ({processor.metrics['damage_density_percentage']}% Area)", "gray")
        ]

        for i, (img, title, cmap) in enumerate(panels):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            if cmap:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=12, fontweight='semibold')
            ax.axis('off')

        # Hide the 6th subplot cell or use it for text summary metrics
        ax_summary = axes[1, 2]
        ax_summary.axis('off')
        summary_text = (
            f"--- QUANTITATIVE METRICS ---\n\n"
            f"• Text Density: {processor.metrics['text_density_percentage']}%\n"
            f"• Surface Damage: {processor.metrics['damage_density_percentage']}%\n"
            f"• Total Resolution: {processor.metrics['total_pixels']:,} px\n"
            f"• Status: Successfully Processed"
        )
        ax_summary.text(0.05, 0.5, summary_text, fontsize=11, family='monospace', 
                        bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.9))

        plt.tight_layout()
        output_path = os.path.join(output_dir, "analysis_report.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Analysis report successfully saved to {output_path}")
        plt.show()