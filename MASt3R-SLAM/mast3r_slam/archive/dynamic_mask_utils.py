import torch
import os
import numpy as np
import cv2
import lietorch
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mast3r_slam.mast3r_utils import mast3r_asymmetric_inference, decoder
from thirdparty.monst3r.third_party.raft import load_RAFT
from thirdparty.monst3r.dust3r.utils.goem_opt import DepthBasedWarping
from thirdparty.monst3r.third_party.sam2.sam2.build_sam import build_sam2_video_predictor
from mast3r_slam.mast3r_utils import resize_img

_MONST3R_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_MONST3R_BASE_PATH = os.path.normpath(os.path.join(_MONST3R_UTILS_DIR, "..", "thirdparty", "monst3r"))
SAM2_CHECKPOINT_DEFAULT = os.path.join(_MONST3R_BASE_PATH, "third_party", "sam2", "checkpoints", "sam2.1_hiera_large.pt")
SAM2_MODEL_CONFIG_ABSOLUTE_PATH = os.path.join(_MONST3R_BASE_PATH, "third_party", "sam2", "sam2", "configs", "sam2.1/sam2.1_hiera_l.yaml")
SAM2_MODEL_CONFIG_NAME_FOR_HYDRA = "configs/sam2.1/sam2.1_hiera_l.yaml"

def save_error_map_debug(frame_i, flow_ij, ego_flow_ij, err_map, norm_err_map, threshold):
    """
    Save debug visualizations of the optical flow, ego-motion flow, error map, and thresholded masks.
    
    Args:
        frame_i: Current frame object
        flow_ij: Optical flow from RAFT (2xHxW)
        ego_flow_ij: Ego-motion flow (3xHxW, use first 2 channels)
        err_map: Raw error map (HxW)
        norm_err_map: Normalized error map (HxW)
        threshold: Current threshold value
    """
    try:
        from mast3r_slam.config import config
        
        # Get dataset info for organizing debug output
        dataset_name = config.get("dataset", {}).get("name", "unknown_dataset")
        video_name = config.get("dataset", {}).get("sequence", config.get("dataset", {}).get("video", "unknown_video"))
        frame_id = frame_i.frame_id
        
        # Create save directory
        save_dir = os.path.join("logs", dataset_name, video_name, "debug_error_maps")
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert tensors to numpy
        flow_ij_np = flow_ij.cpu().numpy()  # 2xHxW
        ego_flow_ij_np = ego_flow_ij[:2].cpu().numpy()  # 2xHxW (only flow components)
        err_map_np = err_map.cpu().numpy()  # HxW
        norm_err_map_np = norm_err_map.cpu().numpy()  # HxW
        
        # Create figure with subplots (use constrained_layout to avoid colorbar issues)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
        fig.suptitle(f'Frame {frame_id} - Dynamic Mask Debug (threshold={threshold:.3f})', fontsize=16)
        
        # 1. Original image
        img_np = frame_i.uimg.cpu().numpy()
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 2. Optical flow magnitude
        flow_mag = np.sqrt(flow_ij_np[0]**2 + flow_ij_np[1]**2)
        im1 = axes[0, 1].imshow(flow_mag, cmap='jet')
        axes[0, 1].set_title('Optical Flow Magnitude')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. Ego-motion flow magnitude
        ego_flow_mag = np.sqrt(ego_flow_ij_np[0]**2 + ego_flow_ij_np[1]**2)
        im2 = axes[0, 2].imshow(ego_flow_mag, cmap='jet')
        axes[0, 2].set_title('Ego-Motion Flow Magnitude')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # 4. Flow difference magnitude (this is err_map)
        im3 = axes[0, 3].imshow(err_map_np, cmap='hot')
        axes[0, 3].set_title('Flow Difference (Error Map)')
        axes[0, 3].axis('off')
        plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # 5. Normalized error map
        im4 = axes[1, 0].imshow(norm_err_map_np, cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title('Normalized Error Map')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 6. Threshold visualization (show current threshold level)
        thresholded_mask = norm_err_map_np > threshold
        axes[1, 1].imshow(thresholded_mask, cmap='gray')
        axes[1, 1].set_title(f'Thresholded Mask (>{threshold:.3f})')
        axes[1, 1].axis('off')
        
        # 7. Multiple threshold levels for comparison
        thresholds_to_test = [0.2, 0.35, 0.5, 0.7]
        colors = ['blue', 'green', 'red', 'yellow']
        overlay_img = img_np.copy()
        
        for i, thresh in enumerate(thresholds_to_test):
            mask = norm_err_map_np > thresh
            if mask.any():
                # Create colored overlay
                overlay = np.zeros_like(img_np)
                color = plt.cm.tab10(i)[:3]  # Get RGB color
                overlay[mask] = color
                overlay_img = overlay_img * 0.7 + overlay * 0.3
        
        axes[1, 2].imshow(overlay_img.clip(0, 1))
        axes[1, 2].set_title('Multi-threshold Overlay')
        axes[1, 2].axis('off')
        
        # Add legend for overlay
        legend_elements = [plt.Line2D([0], [0], color=plt.cm.tab10(i)[:3], lw=3, 
                                    label=f'>{thresh:.2f}') for i, thresh in enumerate(thresholds_to_test)]
        axes[1, 2].legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 8. Error map statistics
        axes[1, 3].hist(norm_err_map_np.flatten(), bins=50, alpha=0.7, color='blue', density=True)
        axes[1, 3].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Current threshold: {threshold:.3f}')
        axes[1, 3].set_title('Normalized Error Distribution')
        axes[1, 3].set_xlabel('Normalized Error Value')
        axes[1, 3].set_ylabel('Density')
        axes[1, 3].legend()
        axes[1, 3].grid(True, alpha=0.3)
        
        # Add statistics text
        mean_err = np.mean(norm_err_map_np)
        std_err = np.std(norm_err_map_np)
        max_err = np.max(norm_err_map_np)
        min_err = np.min(norm_err_map_np)
        
        stats_text = f'Stats:\nMean: {mean_err:.3f}\nStd: {std_err:.3f}\nMin: {min_err:.3f}\nMax: {max_err:.3f}'
        axes[1, 3].text(0.02, 0.98, stats_text, transform=axes[1, 3].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Note: Using constrained_layout=True instead of tight_layout() to avoid colorbar issues
        
        # Save the debug plot
        save_path = os.path.join(save_dir, f"frame_{frame_id:06d}_error_debug.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved error map debug visualization: {save_path}")
        print(f"Error statistics - Mean: {mean_err:.3f}, Std: {std_err:.3f}, Current threshold: {threshold:.3f}")
        
        # Also save raw error map as numpy for further analysis
        np_save_path = os.path.join(save_dir, f"frame_{frame_id:06d}_norm_error_map.npy")
        np.save(np_save_path, norm_err_map_np)
        
    except Exception as e:
        print(f"Error in save_error_map_debug: {e}")
        import traceback
        traceback.print_exc()

def analyze_threshold_performance(dataset_name="unknown_dataset", video_name="unknown_video", frame_range=None):
    """
    Analyze the performance of different thresholds across saved error maps.
    This function loads saved normalized error maps and provides recommendations for optimal thresholds.
    
    Args:
        dataset_name: Name of the dataset
        video_name: Name of the video sequence
        frame_range: Optional tuple (start_frame, end_frame) to limit analysis
        
    Returns:
        dict: Analysis results with recommended thresholds
    """
    try:
        import glob
        
        # Load all saved error maps
        save_dir = os.path.join("logs", dataset_name, video_name, "debug_error_maps")
        if not os.path.exists(save_dir):
            print(f"No error maps found in {save_dir}")
            return None
            
        error_map_files = glob.glob(os.path.join(save_dir, "*_norm_error_map.npy"))
        
        if not error_map_files:
            print(f"No error map files found in {save_dir}")
            return None
            
        print(f"Found {len(error_map_files)} error map files for analysis")
        
        # Load and analyze error maps
        all_errors = []
        frame_stats = []
        
        for file_path in sorted(error_map_files):
            try:
                error_map = np.load(file_path)
                all_errors.append(error_map.flatten())
                
                # Extract frame number
                filename = os.path.basename(file_path)
                frame_num = int(filename.split('_')[1])
                
                # Skip if outside frame range
                if frame_range and (frame_num < frame_range[0] or frame_num > frame_range[1]):
                    continue
                    
                # Compute statistics for this frame
                stats = {
                    'frame': frame_num,
                    'mean': np.mean(error_map),
                    'std': np.std(error_map),
                    'median': np.median(error_map),
                    'p75': np.percentile(error_map, 75),
                    'p90': np.percentile(error_map, 90),
                    'p95': np.percentile(error_map, 95),
                    'p99': np.percentile(error_map, 99),
                    'max': np.max(error_map)
                }
                frame_stats.append(stats)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not frame_stats:
            print("No valid error maps could be loaded")
            return None
            
        # Combine all error values
        combined_errors = np.concatenate(all_errors)
        
        # Overall statistics
        overall_stats = {
            'mean': np.mean(combined_errors),
            'std': np.std(combined_errors),
            'median': np.median(combined_errors),
            'p75': np.percentile(combined_errors, 75),
            'p90': np.percentile(combined_errors, 90),
            'p95': np.percentile(combined_errors, 95),
            'p99': np.percentile(combined_errors, 99),
            'max': np.max(combined_errors)
        }
        
        # Threshold recommendations
        recommendations = {
            'conservative': overall_stats['p90'],  # Keep only top 10% as dynamic
            'moderate': overall_stats['p75'],      # Keep top 25% as dynamic
            'aggressive': overall_stats['median'] + overall_stats['std'],  # Mean + 1 std
            'very_aggressive': overall_stats['mean'] + 0.5 * overall_stats['std']  # Mean + 0.5 std
        }
        
        # Print analysis
        print("\n" + "="*60)
        print("THRESHOLD ANALYSIS RESULTS")
        print("="*60)
        print(f"Analyzed {len(frame_stats)} frames")
        print(f"Total pixels analyzed: {len(combined_errors):,}")
        
        print(f"\nOverall Error Statistics:")
        print(f"  Mean:   {overall_stats['mean']:.4f}")
        print(f"  Median: {overall_stats['median']:.4f}")
        print(f"  Std:    {overall_stats['std']:.4f}")
        print(f"  75th percentile: {overall_stats['p75']:.4f}")
        print(f"  90th percentile: {overall_stats['p90']:.4f}")
        print(f"  95th percentile: {overall_stats['p95']:.4f}")
        print(f"  99th percentile: {overall_stats['p99']:.4f}")
        print(f"  Max:    {overall_stats['max']:.4f}")
        
        print(f"\nRecommended Thresholds:")
        print(f"  Conservative (top 10%):     {recommendations['conservative']:.4f}")
        print(f"  Moderate (top 25%):         {recommendations['moderate']:.4f}")
        print(f"  Aggressive (mean+std):      {recommendations['aggressive']:.4f}")
        print(f"  Very Aggressive (mean+0.5std): {recommendations['very_aggressive']:.4f}")
        
        print(f"\nCurrent default threshold: 0.35")
        current_performance = (combined_errors > 0.35).sum() / len(combined_errors) * 100
        print(f"  -> Marks {current_performance:.2f}% of pixels as dynamic")
        
        print("\nThreshold Performance Analysis:")
        test_thresholds = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
        for thresh in test_thresholds:
            percentage = (combined_errors > thresh).sum() / len(combined_errors) * 100
            print(f"  Threshold {thresh:.2f}: {percentage:.2f}% of pixels marked as dynamic")
        
        print("="*60)
        
        return {
            'overall_stats': overall_stats,
            'frame_stats': frame_stats,
            'recommendations': recommendations,
            'combined_errors': combined_errors
        }
        
    except Exception as e:
        print(f"Error in threshold analysis: {e}")
        import traceback
        traceback.print_exc()
        return None



@torch.inference_mode()
def get_dynamic_mask(mast3r, raft_model, frame_i, frame_j, threshold=0.35, refine_with_sam2=True, sam2_predictor=None, only_dynamic_points=True):
    """
    Extract point prompts for dynamic objects by analyzing flow differences between two frames.
    
    This function computes optical flow vs ego-motion flow difference and extracts point coordinates
    from regions where the error exceeds the threshold. These points are then used by the inpainting
    pipeline to generate final masks and perform inpainting.
    
    Args:
        mast3r: Initialized MASt3R model.
        raft_model: Initialized RAFT model.
        frame_i: First frame object (needs img, T_WC, K).
        frame_j: Second frame object (needs img, T_WC, K).
        threshold: Threshold for classifying pixels as dynamic based on normalized flow error (0.0-1.0).
        refine_with_sam2: Boolean flag to enable SAM2 refinement (currently unused).
        sam2_predictor: Initialized SAM2 predictor instance (currently unused).
        
    Returns:
        dynamic_mask: Binary mask (HxW tensor) where True indicates dynamic content (for debugging).
        point_prompts: List of (x,y) coordinates for dynamic object points (main output for inpainting).
    """
    device = frame_i.img.device
    h, w = frame_i.img.shape[-2:]
    empty_mask = torch.zeros((h, w), dtype=torch.bool, device=device)

    # 1. Check for calibration (required for ego-motion flow)
    if not hasattr(frame_i, 'K') or not hasattr(frame_j, 'K') or frame_i.K is None or frame_j.K is None:
        print("Warning: Cannot compute dynamic mask without camera calibration (K).")
        return empty_mask, []
    

    # 2. Compute Optical Flow (RAFT)
    try:
        # Prepare images for RAFT (needs BCHW format, scaled to [0, 255])
        img_i = frame_i.img 
        img_j = frame_j.img 

        # Ensure both tensors have the batch dimension (BCHW)
        if img_i.dim() == 3: img_i = img_i.unsqueeze(0)
        if img_j.dim() == 3: img_j = img_j.unsqueeze(0)

        # Rescale from [-1, 1] to [0, 255]
        img_i_raft = (img_i * 0.5 + 0.5) * 255.0
        img_j_raft = (img_j * 0.5 + 0.5) * 255.0

        flow_ij = raft_model(img_i_raft, img_j_raft, iters=20, test_mode=True)[1]
        # Output is Bx2xHxW
        flow_ij = flow_ij.squeeze(0) # Remove batch dim -> 2xHxW
    except Exception as e:
        print(f"Error computing optical flow: {e}")
        return empty_mask, []

    # 3. Get Relative Pose T_ji (transform points from i's frame to j's frame)
    T_WC_i = frame_i.T_WC
    T_WC_j = frame_j.T_WC
    if isinstance(T_WC_i, torch.Tensor): T_WC_i = lietorch.Sim3(T_WC_i)
    if isinstance(T_WC_j, torch.Tensor): T_WC_j = lietorch.Sim3(T_WC_j)

    # Assert that T_WC_i and T_WC_j are Sim3 objects after potential conversion
    assert isinstance(T_WC_i, lietorch.Sim3), f"T_WC_i is not a Sim3 object, but type {type(T_WC_i)}"
    assert isinstance(T_WC_j, lietorch.Sim3), f"T_WC_j is not a Sim3 object, but type {type(T_WC_j)}"

    T_ji = T_WC_j.inv() * T_WC_i

    # 4. Get Depth Map for frame_i using MASt3R
    try:
        if frame_i.feat is None:
             frame_i.feat, frame_i.pos, _ = mast3r._encode_image(frame_i.img, frame_i.img_true_shape)
        
        # Perform mono inference to get depth using MASt3R
        res_i, _ = decoder(mast3r, frame_i.feat, frame_i.feat, frame_i.pos, frame_i.pos, frame_i.img_true_shape, frame_i.img_true_shape)
        depth_i = res_i['pts3d'][0, ..., 2] # HxW
        # DepthBasedWarping expects inverse depth (disparity) -> Bx1xHxW
        inv_depth_i = (1.0 / (depth_i + 1e-6)).unsqueeze(0).unsqueeze(0) # 1x1xHxW
    except Exception as e:
        print(f"Error computing depth map for frame_i: {e}")
        return empty_mask, []

    # 5. Get Intrinsics
    K_i = frame_i.K.unsqueeze(0) # 1x3x3
    K_j = frame_j.K.unsqueeze(0) # 1x3x3
    try:
        inv_K_i = torch.linalg.inv(K_i) # 1x3x3
    except Exception as e:
        print(f"Error inverting intrinsics K_i: {e}")
        return empty_mask, []

    # 6. Compute Ego-Motion Flow
    try:
        depth_warper = DepthBasedWarping().to(device)
        
        # Use world poses from the Sim3 matrices
        R1 = T_WC_i.matrix()[..., :3, :3]
        T1 = T_WC_i.matrix()[..., :3, 3].unsqueeze(-1)
        R2 = T_WC_j.matrix()[..., :3, :3]
        T2 = T_WC_j.matrix()[..., :3, 3].unsqueeze(-1)

        ego_flow_ij, _ = depth_warper(R1, T1, R2, T2, inv_depth_i, K_j, inv_K_i)
        ego_flow_ij = ego_flow_ij.squeeze(0) # Remove batch dim -> 3xHxW (flow_x, flow_y, mask)
    except Exception as e:
        print(f"Error computing ego-motion flow: {e}")
        if 'depth_warper' in locals() and depth_warper is not None:
            del depth_warper
        return empty_mask, []
    finally:
        if 'depth_warper' in locals() and depth_warper is not None:
            del depth_warper

    # 7. Compute Error Map
    # Use only the flow components (first 2 channels)
    flow_diff = flow_ij - ego_flow_ij[:2, ...]
    err_map = torch.norm(flow_diff, dim=0) # HxW

    # 8. Normalize and Threshold
    min_err = torch.min(err_map)
    max_err = torch.max(err_map)
    if max_err > min_err:
        norm_err_map = (err_map - min_err) / (max_err - min_err)
    else:
        norm_err_map = torch.zeros_like(err_map) # Avoid division by zero if error is constant
    
    # Debug: Save error map visualization
    try:
        from mast3r_slam.config import config
        if config.get("debug_save_error_maps", True):
            save_error_map_debug(frame_i, flow_ij, ego_flow_ij, err_map, norm_err_map, threshold)
    except Exception as e:
        print(f"Error saving debug visualizations: {e}")
    
    # Use single threshold from config
    dynamic_mask = norm_err_map > threshold
    print(f"Frame {frame_i.frame_id}: Dynamic mask created with threshold {threshold:.3f}, {dynamic_mask.sum().item()} pixels marked as dynamic")

    # Extract better point prompts from the refined dynamic mask
    dynamic_mask_np = dynamic_mask.cpu().numpy()
    labeled = label(dynamic_mask_np)
    regions = regionprops(labeled)
    point_prompts = []
    min_area = 30  # Ignore tiny regions
    
    for region in regions:
        if region.area < min_area:
            continue
            
        # For large regions, use multiple point prompts for better segmentation
        if region.area > 300:
            # Get region bounding box (min_row, min_col, max_row, max_col)
            min_y, min_x, max_y, max_x = region.bbox
            
            # Calculate dimensions
            width = max_x - min_x
            height = max_y - min_y
            
            # Generate a grid of points for large objects
            num_points = min(5, max(2, region.area // 200))
            
            # Generate a central point
            center_y, center_x = region.centroid
            point_prompts.append((int(center_x), int(center_y)))
            
            # Add additional points in a pattern around the center
            if num_points >= 3:
                # Add points at 1/3 and 2/3 of the width/height
                offsets = [0.33, 0.66]
                for off_x in offsets:
                    for off_y in offsets:
                        x = int(min_x + width * off_x)
                        y = int(min_y + height * off_y)
                        # Check if this point is actually in the mask
                        if dynamic_mask_np[y, x]:
                            point_prompts.append((x, y))
        else:
            # For smaller regions, just use the centroid
            y, x = region.centroid
            point_prompts.append((int(x), int(y)))
    
    # Remove duplicate points
    point_prompts = list(set(point_prompts))
    
    ## 9. SAM2 Refinement
    #if refine_with_sam2 and sam2_predictor is not None and dynamic_mask.any() and len(point_prompts) > 0:
    #    print(f"Attempting SAM2 refinement for frame {frame_i.frame_id}...")
    #    sam2_predictor.eval()
    #    refined_successfully = False
    #    try:
    #        # Prepare image for SAM2
    #        img_sam_np = frame_i.uimg.cpu().numpy()
    #        SAM2_INPUT_SIZE = 512
    #        img_resized_sam = resize_img(img_sam_np, SAM2_INPUT_SIZE)["unnormalized_img"]
    #        h_sam, w_sam = img_resized_sam.shape[:2]
    #        # Convert to CHW tensor [0, 1] for SAM2 state init
    #        img_tensor_sam = torch.from_numpy(img_resized_sam).permute(2, 0, 1).float().to(device) / 255.0
    #        img_tensor_sam = img_tensor_sam.unsqueeze(0) # 1xCxHxW
    #        # Prepare point prompts (Nx2 tensor) in (x,y) order
    #        points_sam = torch.tensor(point_prompts, dtype=torch.float, device=device).unsqueeze(0) # 1xNx2
    #        labels_sam = torch.ones(points_sam.shape[1], dtype=torch.int, device=device).unsqueeze(0) # 1xN (all foreground)
    #        with torch.no_grad():
    #            sam2_refined_mask = None
    #            state = sam2_predictor.init_state(video_path=img_tensor_sam)
    #            sam2_predictor.add_new_points(state, frame_idx=0, obj_id=1, points=points_sam, labels=labels_sam)
    #            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(state, start_frame_idx=0):
    #                if out_frame_idx == 0 and 1 in out_obj_ids:
    #                    obj_idx = out_obj_ids.index(1)
    #                    sam2_refined_mask = (out_mask_logits[obj_idx] > 0.0)
    #                    if sam2_refined_mask.shape[0] == 1:
    #                         sam2_refined_mask = sam2_refined_mask.squeeze(0)
    #                    break
    #        if sam2_refined_mask is not None:
    #            # Resize SAM2 mask back to original frame dimensions (h, w)
    #            sam2_refined_mask_np = sam2_refined_mask.cpu().numpy().astype(np.uint8)
    #            sam2_refined_mask_resized_np = cv2.resize(sam2_refined_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
    #            sam2_refined_mask_resized = torch.from_numpy(sam2_refined_mask_resized_np).to(device=device, dtype=torch.bool)
    #            
    #            if sam2_refined_mask_resized.shape != dynamic_mask.shape:
    #                print(f"Warning: Resized SAM2 mask shape {sam2_refined_mask_resized.shape} mismatch original {dynamic_mask.shape} for frame {frame_i.frame_id}. Discarding SAM2 output.")
    #            else:
    #                original_sum = dynamic_mask.sum().item()
    #                refined_sum = sam2_refined_mask_resized.sum().item()
    #                dynamic_mask = sam2_refined_mask_resized
    #                print(f"Dynamic mask for frame {frame_i.frame_id} refined with SAM2. Original sum: {original_sum}, Refined sum: {refined_sum}")
    #                refined_successfully = True
    #    except Exception as e_sam:
    #        print(f"Error during SAM2 refinement for frame {frame_i.frame_id}: {e_sam}")

    return dynamic_mask, point_prompts 