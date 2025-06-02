#!/usr/bin/env python3

"""
Test script for inpainting integration in MASt3R-SLAM.
This script tests the inpainting pipeline independently and with a sample frame.
"""

import torch
import numpy as np
import PIL.Image
import os

from mast3r_slam.archive.inpainting_utils import InpaintingPipeline

def test_inpainting_pipeline():
    """Test the inpainting pipeline with synthetic data."""
    print("Testing Inpainting Pipeline...")
    
    # Create a synthetic test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create test point prompts (some sample coordinates)
    point_prompts = [(320, 240), (200, 150), (450, 350)]
    
    try:
        # Initialize pipeline
        pipeline = InpaintingPipeline(device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Test point-based inpainting
        print("Testing point-based inpainting...")
        inpainted_img, inpaint_mask = pipeline.inpaint_frame_with_points(
            test_img, point_prompts, dilate_kernel_size=15
        )
        
        print(f"Original image shape: {test_img.shape}")
        print(f"Inpainted image shape: {inpainted_img.shape}")
        print(f"Inpaint mask shape: {inpaint_mask.shape}")
        print(f"Number of inpainted pixels: {inpaint_mask.sum()}")
        
        # Test mask-based inpainting
        print("Testing mask-based inpainting...")
        test_mask = np.zeros((480, 640), dtype=bool)
        test_mask[200:280, 300:380] = True  # Create a rectangular mask
        
        inpainted_img_mask = pipeline.inpaint_frame_with_mask(
            test_img, test_mask, dilate_kernel_size=15
        )
        
        print(f"Mask-based inpainted image shape: {inpainted_img_mask.shape}")
        
        # Save test results
        os.makedirs("test_inpainting_results", exist_ok=True)
        
        PIL.Image.fromarray(test_img).save("test_inpainting_results/original.png")
        PIL.Image.fromarray(inpainted_img).save("test_inpainting_results/inpainted_points.png")
        PIL.Image.fromarray(inpainted_img_mask).save("test_inpainting_results/inpainted_mask.png")
        
        # Create visualization
        mask_overlay = test_img.copy()
        mask_overlay[inpaint_mask] = [255, 0, 0]  # Red overlay for inpainted regions
        PIL.Image.fromarray(mask_overlay).save("test_inpainting_results/mask_overlay.png")
        
        print("Inpainting pipeline test completed successfully!")
        print("Results saved to test_inpainting_results/")
        
        return True
        
    except Exception as e:
        print(f"Inpainting pipeline test failed: {e}")
        return False

def test_integration_with_config():
    """Test integration with config system."""
    print("\nTesting configuration integration...")
    
    try:
        from mast3r_slam.config import config, load_config
        
        # Load base config
        load_config("config/base.yaml")
        
        # Check if inpainting options are available
        inpainting_enabled = config.get("use_inpainting", False)
        sam_model = config.get("inpainting_sam_model", "vit_h")
        dilate_kernel = config.get("inpainting_dilate_kernel", 15)
        
        print(f"Inpainting enabled: {inpainting_enabled}")
        print(f"SAM model: {sam_model}")
        print(f"Dilate kernel size: {dilate_kernel}")
        
        print("Configuration integration test passed!")
        return True
        
    except Exception as e:
        print(f"Configuration integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MASt3R-SLAM Inpainting Integration Test")
    print("=" * 60)
    
    # Test inpainting pipeline
    pipeline_success = test_inpainting_pipeline()
    
    # Test config integration
    config_success = test_integration_with_config()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Inpainting Pipeline: {'PASS' if pipeline_success else 'FAIL'}")
    print(f"Config Integration: {'PASS' if config_success else 'FAIL'}")
    
    if pipeline_success and config_success:
        print("\nAll tests passed! Inpainting integration is ready.")
        print("\nTo use inpainting in your SLAM system:")
        print("1. Make sure the required model checkpoints are available:")
        print("   - SAM checkpoint: thirdparty/inpaint/segment_anything/checkpoints/sam_vit_h_4b8939.pth")
        print("   - LAMA checkpoint: thirdparty/inpaint/pretrained_models/big-lama")
        print("2. Set 'use_inpainting: True' in your config")
        print("3. Optionally enable debug visualization with 'debug_save_inpainting: True'")
    else:
        print("\nSome tests failed. Please check the error messages above.")
    
    print("=" * 60) 