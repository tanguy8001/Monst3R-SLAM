import torch
import numpy as np
import sys
import os
from pathlib import Path
import PIL.Image

# Add the inpaint directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
inpaint_dir = os.path.join(current_dir, "../thirdparty/inpaint")
sys.path.insert(0, inpaint_dir)

from sam_segment import predict_masks_with_sam, build_sam_model
from lama_inpaint import inpaint_img_with_lama, build_lama_model, inpaint_img_with_builded_lama

# Import dilate_mask with hardcoded path to avoid conflicts
import importlib.util
_utils_path = Path(__file__).resolve().parent / "../thirdparty/inpaint/utils/utils.py"
_utils_spec = importlib.util.spec_from_file_location("inpaint_utils_funcs", _utils_path)
_utils_module = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_utils_module)
dilate_mask = _utils_module.dilate_mask

class InpaintingPipeline:
    """
    A pipeline that combines SAM segmentation and LAMA inpainting for dynamic object removal.
    """
    
    def __init__(self, 
                 sam_model_type="vit_b",
                 sam_ckpt="thirdparty/inpaint/pretrained_models/sam_vit_b_01ec64.pth",
                 lama_config="thirdparty/inpaint/lama/configs/prediction/default.yaml", 
                 lama_ckpt="thirdparty/inpaint/pretrained_models/big-lama",
                 device="cuda"):
        """
        Initialize the inpainting pipeline.
        
        Args:
            sam_model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            sam_ckpt: Path to SAM checkpoint
            lama_config: Path to LAMA config file
            lama_ckpt: Path to LAMA checkpoint directory
            device: Device to run models on
        """
        self.device = device
        self.sam_model_type = sam_model_type
        self.sam_ckpt = sam_ckpt
        self.lama_config = lama_config
        self.lama_ckpt = lama_ckpt

        self.sam_predictor = None
        self.lama_model = None
        self._init_models()
    
    def _init_models(self):
        """Initialize SAM and LAMA models."""
        try:
            print(f"Initializing SAM model {self.sam_ckpt}...")
            self.sam_predictor = build_sam_model(
                model_type=self.sam_model_type,
                ckpt_p=self.sam_ckpt,
                device=self.device
            )
            print("SAM model initialized successfully.")
            
            print("Initializing LAMA model...")
            self.lama_model = build_lama_model(
                config_p=self.lama_config,
                ckpt_p=self.lama_ckpt,
                device=self.device
            )
            print("LAMA model initialized successfully.")
            
        except Exception as e:
            print(f"Error initializing inpainting models: {e}")
            print("Inpainting will be disabled.")
            self.sam_predictor = None
            self.lama_model = None
    
    def inpaint_frame_with_points(self, frame_img, point_prompts, dilate_kernel_size=15, debug_save=False, frame_id=None, dataset_name="unknown_dataset", video_name="unknown_video"):
        """
        Inpaint dynamic regions in a frame using point prompts.
        
        Args:
            frame_img: Input image as numpy array (H, W, 3) in [0, 255] range
            point_prompts: List of (x, y) coordinates marking dynamic objects
            dilate_kernel_size: Size of dilation kernel for masks
            debug_save: Whether to save debug visualizations of SAM masks
            frame_id: Current frame ID for saving debug images
            dataset_name: Name of dataset for organizing debug output
            video_name: Name of video sequence for organizing debug output
            
        Returns:
            inpainted_img: Inpainted image as numpy array (H, W, 3) in [0, 255] range
            combined_mask: Combined binary mask of all inpainted regions
            sam_masks: Original SAM-generated masks before dilation (for visualization)
        """
        if self.sam_predictor is None or self.lama_model is None:
            print("Warning: Inpainting models not available. Returning original image.")
            return frame_img, np.zeros((frame_img.shape[0], frame_img.shape[1]), dtype=bool), None
        
        if not point_prompts:
            print("No point prompts provided. Returning original image.")
            return frame_img, np.zeros((frame_img.shape[0], frame_img.shape[1]), dtype=bool), None
        
        # Convert point prompts to required format
        point_coords = [[float(x), float(y)] for x, y in point_prompts]
        point_labels = [1] * len(point_prompts)  # All points are foreground
        
        # Generate masks using SAM
        masks, scores, logits = predict_masks_with_sam(
            frame_img,
            point_coords,
            point_labels,
            model_type=self.sam_model_type,
            ckpt_p=self.sam_ckpt,
            device=self.device
        )
        
        if len(masks) == 0:
            print("No masks generated. Returning original image.")
            return frame_img, np.zeros((frame_img.shape[0], frame_img.shape[1]), dtype=bool), None
        
        # Save the original SAM masks for visualization
        original_sam_masks = masks.copy()
        
        ## Save visualization of SAM masks if debug_save is enabled
        #if debug_save and frame_id is not None:
        #    try:
        #        # Create a visualization of the masks
        #        img_pil = PIL.Image.fromarray(frame_img)
        #        img_pil = img_pil.convert("RGBA")
        #        overlay = PIL.Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        #        
        #        # Combine all masks with different colors for visualization
        #        colors = [(255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128), (255, 255, 0, 128), (255, 0, 255, 128)]
        #        
        #        for i, mask in enumerate(original_sam_masks):
        #            mask_color = colors[i % len(colors)]
        #            mask_pixels = np.zeros((*mask.shape, 4), dtype=np.uint8)
        #            mask_pixels[mask] = mask_color
        #            mask_image = PIL.Image.fromarray(mask_pixels, 'RGBA')
        #            overlay.paste(mask_image, (0, 0), mask_image)
        #        
        #        img_with_masks = PIL.Image.alpha_composite(img_pil, overlay)
        #        img_with_masks = img_with_masks.convert("RGB")
        #        
        #        # Save the visualization
        #        save_dir = os.path.join("logs", dataset_name, video_name, "debug_sam_masks")
        #        os.makedirs(save_dir, exist_ok=True)
        #        save_path = os.path.join(save_dir, f"frame_{frame_id:06d}_sam_masks.png")
        #        img_with_masks.save(save_path)
        #    except Exception as e:
        #        print(f"Error saving SAM masks visualization: {e}")
        
        # Convert masks to uint8 and dilate
        masks = masks.astype(np.uint8) * 255
        if dilate_kernel_size is not None:
            masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
        
        # Combine all masks
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask)
        
        # Inpaint using LAMA
        inpainted_img = inpaint_img_with_builded_lama(
            model=self.lama_model,
            img=frame_img,
            mask=combined_mask,
            device=self.device
        )
        
        # Convert combined mask to boolean
        combined_mask_bool = (combined_mask > 127).astype(bool)
        
        return inpainted_img, combined_mask_bool, original_sam_masks