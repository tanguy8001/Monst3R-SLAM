# Inpainting Integration for MASt3R-SLAM

This document describes the integration of inpainting capabilities into MASt3R-SLAM for improved dynamic object handling.

## Overview

The inpainting integration allows MASt3R-SLAM to intelligently fill in dynamic regions detected by MonST3R instead of simply masking them with black pixels. This preserves image structure and improves matching performance in dynamic environments.

## Key Components

### 1. InpaintingPipeline (`mast3r_slam/inpainting_utils.py`)

A wrapper class that combines:
- **SAM (Segment Anything Model)**: Generates precise masks from point prompts
- **LAMA (Large Mask Inpainting)**: Performs high-quality inpainting

#### Features:
- Point prompt-based inpainting (primary method)
- Mask-based inpainting (fallback method)
- Automatic model loading and initialization
- Graceful error handling

### 2. Modified Tracker (`mast3r_slam/tracker2.py`)

Enhanced tracking pipeline with:
- Automatic inpainting pipeline initialization
- Three-tier fallback strategy:
  1. **Point-based inpainting** (best quality)
  2. **Mask-based inpainting** (fallback)
  3. **Black masking** (original behavior)

### 3. Enhanced Dynamic Mask Detection (`mast3r_slam/monst3r_utils.py`)

`get_dynamic_mask()` now always returns point prompts extracted from dynamic regions using connected component analysis.

## Configuration Options

Add these options to your config YAML files:

```yaml
# Inpainting configuration
use_inpainting: True                    # Enable/disable inpainting
inpainting_dilate_kernel: 15           # Dilation kernel size for masks
inpainting_sam_model: "vit_h"          # SAM model type
inpainting_sam_ckpt: "path/to/sam.pth" # SAM checkpoint path
inpainting_lama_config: "path/to/lama_config.yaml"  # LAMA config
inpainting_lama_ckpt: "path/to/lama"   # LAMA checkpoint directory

# Debug options
debug_save_inpainting: False           # Save before/after inpainting images
debug_save_dynamic_mask: False         # Save dynamic mask overlays
debug_save_final_valid_opt_mask: False # Save final optimization masks
```

## Setup Instructions

### 1. Install Dependencies

The inpainting system requires additional dependencies:

```bash
# Install SAM dependencies (if not already installed)
pip install segment-anything

# Install LAMA dependencies
pip install omegaconf
pip install albumentations
pip install kornia
```

### 2. Download Model Checkpoints

#### SAM Checkpoints:
```bash
mkdir -p thirdparty/inpaint/segment_anything/checkpoints/
cd thirdparty/inpaint/segment_anything/checkpoints/

# Download SAM ViT-H checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### LAMA Checkpoints:
```bash
mkdir -p thirdparty/inpaint/pretrained_models/
cd thirdparty/inpaint/pretrained_models/

# Download and extract big-lama model
wget https://github.com/saic-mdal/lama/releases/download/v1.0/big-lama.zip
unzip big-lama.zip
```

### 3. Verify Installation

Run the test script to verify everything is working:

```bash
python test_inpainting_integration.py
```

## Usage

### Basic Usage

Simply enable inpainting in your config:

```yaml
use_dynamic_mask: True
use_inpainting: True
```

### Advanced Configuration

For fine-tuning, adjust these parameters:

```yaml
# Dynamic mask detection
dynamic_mask_threshold: 0.35        # Sensitivity for dynamic detection
refine_dynamic_mask_with_sam2: True # Use SAM2 for mask refinement

# Inpainting parameters
inpainting_dilate_kernel: 15        # Larger = more conservative inpainting
inpainting_sam_model: "vit_h"       # vit_h (best), vit_l, vit_b (faster)

# Debug visualization
debug_save_inpainting: True         # Enable to see inpainting results
```

## How It Works

### Pipeline Flow

1. **Dynamic Detection**: MonST3R computes optical flow and detects dynamic regions
2. **Point Extraction**: Connected component analysis extracts representative points
3. **Segmentation**: SAM generates precise masks from the point prompts
4. **Inpainting**: LAMA fills in the masked regions with plausible content
5. **Matching**: MASt3R performs feature matching on the inpainted image

### Fallback Strategy

```
Point Prompts Available? 
├─ Yes → SAM + LAMA Inpainting (Best Quality)
└─ No → Direct Mask Inpainting (Good Quality)
   └─ Inpainting Failed? → Black Masking (Original Behavior)
```

## Debugging and Visualization

Enable debug options to visualize the process:

```yaml
debug_save_inpainting: True
debug_save_dynamic_mask: True
```

This creates debug images in `logs/{dataset}/{sequence}/`:
- `debug_inpainting/`: Before/after inpainting comparisons
- `debug_dynamic_mask/`: Dynamic mask overlays
- `debug_final_valid_opt_mask/`: Final optimization masks

## Performance Considerations

### Memory Usage
- SAM + LAMA models require ~2-3GB additional GPU memory
- Consider using smaller SAM models (`vit_l` or `vit_b`) if memory is limited

### Speed
- Inpainting adds ~200-500ms per frame depending on:
  - Number of dynamic regions
  - SAM model size
  - Image resolution
- For real-time applications, consider:
  - Using `vit_b` SAM model
  - Reducing `inpainting_dilate_kernel`
  - Setting `use_inpainting: False` for very time-critical applications

## Troubleshooting

### Common Issues

1. **"SAM model not found"**
   - Ensure SAM checkpoint is downloaded to the correct path
   - Check file permissions

2. **"LAMA model not found"**
   - Verify LAMA checkpoint directory structure
   - Ensure `config.yaml` exists in the LAMA directory

3. **Out of memory errors**
   - Reduce SAM model size: `inpainting_sam_model: "vit_b"`
   - Reduce image resolution in dataset config

4. **Poor inpainting quality**
   - Increase `inpainting_dilate_kernel` for larger inpainted regions
   - Check if dynamic masks are too large/small by enabling debug visualization

### Performance Optimization

```yaml
# For speed-optimized setup:
use_inpainting: True
inpainting_sam_model: "vit_b"        # Faster but less accurate
inpainting_dilate_kernel: 10         # Smaller kernel = faster
refine_dynamic_mask_with_sam2: False # Skip SAM2 refinement
```

```yaml
# For quality-optimized setup:
use_inpainting: True
inpainting_sam_model: "vit_h"        # Best quality
inpainting_dilate_kernel: 20         # More conservative inpainting
refine_dynamic_mask_with_sam2: True  # Use SAM2 refinement
```

## Expected Improvements

With inpainting enabled, you should see:

1. **Better matching**: More feature correspondences in scenes with dynamic objects
2. **Improved tracking**: More stable camera pose estimation
3. **Reduced drift**: Better long-term consistency
4. **Cleaner pointmaps**: Less noise from masked regions

## File Structure

```
mast3r_slam/
├── inpainting_utils.py          # Main inpainting pipeline
├── tracker2.py                  # Modified tracker with inpainting
├── monst3r_utils.py            # Enhanced dynamic mask detection
config/
├── base.yaml                   # Base config with inpainting options
test_inpainting_integration.py  # Test script
thirdparty/inpaint/            # Inpainting models and utilities
└── pretrained_models/         # Model checkpoints
```

## Citation

If you use this inpainting integration in your research, please cite the original LAMA and SAM papers along with MASt3R-SLAM.

---

For questions or issues, please refer to the main MASt3R-SLAM documentation or create an issue in the repository. 