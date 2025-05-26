# Configuration Structure for MASt3R-SLAM with Inpainting

This document explains the clean configuration structure and how to avoid conflicts between different config files.

## Configuration Hierarchy

```
base.yaml (Base configuration)
├── calib.yaml (Calibrated camera mode)
│   └── calib_inpainting_test.yaml (Testing with full debug)
├── eval_calib.yaml (Evaluation mode)
└── eth3d.yaml (ETH3D dataset specific)
```

## File Descriptions

### `config/base.yaml` - Base Configuration

**Purpose**: Default settings that all other configs inherit from.

**Key Features**:
- Inpainting enabled by default (`use_inpainting: True`)
- Dynamic masking disabled by default (`use_dynamic_mask: False`) 
- All debug visualization disabled for performance
- Standard camera mode (`use_calib: False`)

**Usage**: Use directly for basic SLAM without calibration.

### `config/calib.yaml` - Calibrated Camera Mode

**Purpose**: Optimized settings for calibrated cameras.

**Key Overrides**:
- Enables calibrated mode (`use_calib: True`)
- Enables dynamic masking (`use_dynamic_mask: True`)
- Enables SAM2 refinement (`refine_dynamic_mask_with_sam2: True`)
- Enables keyframe dynamic masking (`use_dynamic_mask_for_keyframes: True`)
- Higher subsampling (`subsample: 4`) for better accuracy

**Inherited from base.yaml**:
- All inpainting settings
- All SLAM algorithm parameters
- All debug flags (disabled)

**Usage**: Use for calibrated cameras with known intrinsics.

### `config/calib_inpainting_test.yaml` - Debug Configuration

**Purpose**: Testing configuration with all debug features enabled.

**Key Features**:
- Inherits from `calib.yaml`
- Enables ALL debug visualizations
- Slightly more sensitive dynamic detection (`threshold: 0.3`)
- Lower subsampling (`subsample: 2`) for faster testing

**Usage**: Use for testing and debugging the inpainting integration.

## Configuration Parameters

### Dynamic Processing
```yaml
use_dynamic_mask: false/true          # Enable dynamic object detection
dynamic_mask_threshold: 0.35          # Sensitivity (0.0-1.0, lower = more sensitive)
refine_dynamic_mask_with_sam2: false/true  # Use SAM2 for mask refinement
use_dynamic_mask_for_keyframes: false/true # Advanced keyframe masking
```

### Inpainting Integration
```yaml
use_inpainting: true                  # Enable inpainting pipeline
inpainting_dilate_kernel: 15          # Mask dilation size
inpainting_sam_model: "vit_h"         # SAM model (vit_h, vit_l, vit_b)
inpainting_sam_ckpt: "path/to/sam"    # SAM checkpoint path
inpainting_lama_config: "path/config" # LAMA config path
inpainting_lama_ckpt: "path/to/lama"  # LAMA checkpoint path
```

### Debug Visualization
```yaml
debug_save_dynamic_mask: false       # Save dynamic mask overlays
debug_save_inpainting: false         # Save before/after inpainting
debug_save_final_valid_opt_mask: false # Save optimization masks
debug_save_new_kf_dynamic_mask: false  # Save keyframe dynamic masks
```

## Usage Examples

### 1. Basic SLAM (no calibration)
```bash
python main.py --config config/base.yaml
```

### 2. Calibrated SLAM with inpainting
```bash
python main.py --config config/calib.yaml
```

### 3. Testing/debugging inpainting
```bash
python main.py --config config/calib_inpainting_test.yaml
```

### 4. Quick performance test (no inpainting)
Create a custom config:
```yaml
inherit: "config/base.yaml"
use_inpainting: false
use_dynamic_mask: false
```

## Conflict Resolution

### Potential Conflicts and Solutions

1. **Redundant Parameters**
   - ✅ **Solved**: Removed duplicate parameters between base.yaml and calib.yaml
   - Each parameter is defined only once in the hierarchy

2. **Inpainting Dependencies**
   - ✅ **Solved**: Inpainting settings are defined in base.yaml and inherited
   - Override only if you need different settings

3. **Debug Flag Conflicts**
   - ✅ **Solved**: All debug flags organized in base.yaml
   - Override specific flags in derived configs as needed

4. **Dynamic Masking Logic**
   - ✅ **Solved**: Clear hierarchy: base (disabled) → calib (enabled) → test (full debug)

### Best Practices

1. **Always inherit from base.yaml** unless you have a specific reason not to
2. **Only override parameters you need to change** - don't repeat defaults
3. **Use comments** to explain why you're overriding specific parameters
4. **Test with debug configs** before using production configs
5. **Keep custom configs minimal** - let inheritance do the work

## Performance Recommendations

### For Speed (Real-time applications)
```yaml
inherit: "config/base.yaml"
use_inpainting: true
inpainting_sam_model: "vit_b"        # Faster SAM model
use_dynamic_mask: true
refine_dynamic_mask_with_sam2: false # Skip SAM2 refinement
dataset:
  subsample: 1                       # Lower subsampling
```

### For Quality (Offline processing)
```yaml
inherit: "config/calib.yaml"
inpainting_sam_model: "vit_h"        # Best SAM model
inpainting_dilate_kernel: 20         # More conservative inpainting
refine_dynamic_mask_with_sam2: true  # Use SAM2 refinement
dataset:
  subsample: 4                       # Higher subsampling
```

### For Memory-constrained Systems
```yaml
inherit: "config/base.yaml"
use_inpainting: false                # Disable inpainting to save memory
use_dynamic_mask: true
inpainting_sam_model: "vit_b"        # If you enable inpainting later
```

## Troubleshooting

### Common Issues

1. **"Parameter X defined multiple times"**
   - Check inheritance chain for duplicate definitions
   - Remove redundant parameters from derived configs

2. **"Inpainting not working in calib mode"**
   - Verify `use_inpainting: true` is set (inherited from base.yaml)
   - Check that dynamic masking is enabled (`use_dynamic_mask: true`)

3. **"Debug images not saving"**
   - Verify debug flags are enabled in your config
   - Check that output directories are writable

4. **"Conflicting settings"**
   - Use the test config to verify your setup works
   - Start with a known working config and modify incrementally

---

This structure ensures clean inheritance, avoids conflicts, and provides flexibility for different use cases. 