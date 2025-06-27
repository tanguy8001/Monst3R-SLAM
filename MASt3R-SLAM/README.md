# MonST3R-SLAM: Real-Time Dynamic SLAM with Optical Flow Filtering

A novel SLAM system that integrates MonST3R and MASt3R to achieve robust real-time performance in dynamic environments. Our approach leverages the complementary strengths of these foundation models: MonST3R provides dense 3D pointmaps with confidence estimates, while MASt3R contributes rich feature descriptors and correspondence matching capabilities.

## 🚀 Key Features

- **Dynamic Object Handling**: Novel 3D pointmap-based filtering mechanism that operates directly on 3D representations rather than 2D image masking
- **Optical Flow Analysis**: RAFT-based dynamic detection comparing observed vs. ego-motion flow
- **SAM2 Refinement**: Optional semantic refinement of dynamic masks using SAM2
- **Real-time Performance**: Efficient implementation suitable for real-time applications
- **Foundation Model Integration**: Seamless integration of MonST3R and MASt3R capabilities

## 📋 Prerequisites

- CUDA-capable GPU with at least 8GB VRAM
- Conda/Miniconda
- Python 3.8+

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd MonST3R-SLAM
```

### 2. Setup Base Dependencies

First, follow the official installation instructions for the base repositories:

**MASt3R-SLAM Setup:**
1. Visit the [official MASt3R-SLAM repository](https://github.com/edexheim/MASt3R-SLAM)
2. Follow their installation instructions for setting up the conda environment and downloading checkpoints
3. Ensure all dependencies are properly installed

**MonST3R Setup:**
1. Navigate to `thirdparty/monst3r/`
2. Visit the [official MonST3R repository](https://github.com/Junyi42/monst3r)
3. Follow their installation instructions within the `thirdparty/monst3r/` directory
4. Download the required MonST3R checkpoints

### 3. Additional Dependencies

Install additional dependencies for dynamic object detection:

```bash
# RAFT for optical flow
cd thirdparty/monst3r/third_party/RAFT
# Follow RAFT installation instructions and download pretrained models

# SAM2 for mask refinement (optional)
cd thirdparty/monst3r/third_party/sam2
# Follow SAM2 installation instructions and download checkpoints
```

### 4. Dataset Setup

Place your datasets in the `datasets/` directory following the structure:
```
datasets/
├── tum/
│   ├── rgbd_dataset_freiburg1_desk/
│   └── ...
├── bonn/
│   ├── removing_nonobstructing_box2/
│   └── ...
└── ...
```

## 🎯 Usage

### Basic SLAM Execution

```bash
# Run on TUM dataset
python main_monster_slam.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --config config/base.yaml

# Run on Bonn RGB-D Dynamic dataset
python main_monster_slam.py --dataset datasets/bonn/removing_nonobstructing_box2 --config config/base.yaml

# Run with calibration
python main_monster_slam.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --config config/base.yaml --calib calib.yaml
```

### Configuration Options

Key configuration parameters in `config/base.yaml`:

```yaml
# Dynamic masking settings
use_dynamic_mask: true
dynamic_mask_threshold: 0.35
refine_dynamic_mask_with_sam2: true

# Debugging options
debug_save_dynamic_mask: true
debug_save_pointmap_visualization: false
debug_save_3d_pointclouds: false

# Dataset settings
dataset:
  img_downsample: 4  # Adjust based on VRAM
  subsample: 1
```

## 📁 Project Structure

### Core Implementation Files

- **`main_monster_slam.py`**: Main SLAM pipeline orchestrating MonST3R and MASt3R integration
- **`mast3r_slam/monst3r_utils.py`**: Core utilities for MonST3R integration, dynamic masking, and pointmap filtering
- **`mast3r_slam/tracker2.py`**: Frame tracking with dynamic object handling and pose optimization

### Key Functions in `monst3r_utils.py`

- **`monst3r_asymmetric_inference_with_dynamic_mask()`**: MonST3R inference with dynamic filtering
- **`apply_dynamic_mask_to_pointmaps()`**: 3D pointmap confidence modulation for dynamic regions
- **`get_dynamic_mask()`**: Optical flow-based dynamic detection with SAM2 refinement
- **`save_pointmap_visualization()`**: Debug visualization tools for pointmap analysis

### Archive Files

- **`mast3r_slam/archive/inpainting_utils.py`**: Experimental inpainting-based dynamic object removal
- **`mast3r_slam/archive/dynamic_mask_utils.py`**: Alternative dynamic masking approaches

### Configuration

- **`config/base.yaml`**: Main configuration file with tracking, optimization, and dynamic handling parameters
- **`config/`**: Additional configuration files for different scenarios

### Third-party Dependencies

- **`thirdparty/monst3r/`**: MonST3R submodule for motion-aware 3D reconstruction
- **`thirdparty/Easi3R/`**: Experimental Easi3R integration (incomplete)
- **`thirdparty/inpaint/`**: SAM + LAMA inpainting pipeline components

## 🔬 Methodology

### Dynamic Object Detection

1. **Optical Flow Computation**: Uses RAFT to compute observed optical flow between consecutive frames
2. **Ego-Motion Flow Estimation**: Computes expected flow from camera motion and MonST3R depth
3. **Flow Discrepancy Analysis**: Identifies dynamic regions by comparing observed vs. expected flow
4. **SAM2 Refinement**: Optional semantic segmentation refinement for improved mask quality

### 3D Pointmap Filtering

Instead of traditional 2D masking, our approach:
- Modulates confidence values for dynamic 3D points
- Zeros out descriptors in dynamic regions
- Preserves geometric consistency for foundation model inference
- Maintains temporal coherence through keyframe management

## 📊 Evaluation

The system has been evaluated on:
- **TUM RGB-D Dataset**: Standard SLAM benchmarking
- **Bonn RGB-D Dynamic Dataset**: Dynamic environment evaluation with moving objects

Results show the trade-offs between dynamic filtering and information preservation, particularly in scenarios with large dynamic objects.

## 🐛 Debug Features

Enable debug visualizations to analyze system behavior:

```yaml
debug_save_dynamic_mask: true           # Save dynamic mask overlays
debug_save_pointmap_visualization: true # Save pointmap before/after filtering  
debug_save_final_valid_opt_mask: true   # Save final optimization masks
debug_save_3d_pointclouds: true        # Save 3D point cloud comparisons
```

Debug outputs are saved to `logs/{dataset_name}/{sequence_name}/debug_*/`

## 🔮 Future Work

- **Easi3R Integration**: Complete sliding window approach for global pointmap handling
- **Information-Preserving Filtering**: Alternative approaches that retain more geometric information
- **Temporal Consistency**: Enhanced dynamic object tracking across longer sequences
- **Multi-Object Scenarios**: Improved handling of multiple simultaneous dynamic objects

## 📄 License

This project builds upon MASt3R-SLAM and MonST3R. Please refer to their respective licenses for usage terms.

## 🙏 Acknowledgments

- [MASt3R-SLAM](https://github.com/edexheim/MASt3R-SLAM) for the base SLAM framework
- [MonST3R](https://github.com/Junyi42/monst3r) for motion-aware 3D reconstruction
- [RAFT](https://github.com/princeton-vl/RAFT) for optical flow estimation
- [SAM2](https://github.com/facebookresearch/sam2) for semantic segmentation refinement 