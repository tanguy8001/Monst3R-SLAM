import torch
import os
import numpy as np
import cv2
import lietorch
from skimage.measure import label, regionprops

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


@torch.inference_mode()
def get_dynamic_mask(mast3r, raft_model, frame_i, frame_j, threshold=0.35, refine_with_sam2=True, sam2_predictor=None, only_dynamic_points=True):
    """
    Get dynamic mask between two frames using MASt3R and RAFT.
    Optionally refines the mask using SAM2.
    
    Compares optical flow (RAFT) with ego-motion flow (MASt3R depth + relative pose)
    to identify inconsistencies indicative of dynamic objects.
    
    Args:
        mast3r: Initialized MASt3R model.
        raft_model: Initialized RAFT model.
        frame_i: First frame object (needs img, T_WC, K).
        frame_j: Second frame object (needs img, T_WC, K).
        threshold: Threshold for classifying pixels as dynamic based on normalized flow error (0.0-1.0).
        refine_with_sam2: Boolean flag to enable SAM2 refinement.
        sam2_predictor: Initialized SAM2 predictor instance (required if refine_with_sam2 is True).
        
    Returns:
        dynamic_mask: Binary mask (HxW tensor) where True indicates dynamic content.
                      Returns an empty mask if calculation fails (e.g., missing calibration).
                      Mask is on the same device as frame_i.img.
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

    dynamic_mask = norm_err_map > threshold # HxW boolean tensor, on device

    # Extract point prompts from dynamic mask
    dynamic_mask_np = dynamic_mask.cpu().numpy()
    labeled = label(dynamic_mask_np)
    regions = regionprops(labeled)
    point_prompts = []
    min_area = 20 # Ignore tiny regions
    for region in regions:
        if region.area >= min_area:
            # centroid gives (row, col) i.e., (y, x)
            y, x = region.centroid
            point_prompts.append((int(x), int(y))) # Need (x,y) for inpainting

    # 9. SAM2 Refinement
    if refine_with_sam2 and sam2_predictor is not None and dynamic_mask.any():
        print(f"Attempting SAM2 refinement for frame {frame_i.frame_id}...")
        sam2_predictor.eval()
        refined_successfully = False
        try:
            if len(point_prompts) > 0:
                # Prepare image for SAM2
                img_sam_np = frame_i.uimg.cpu().numpy()
                SAM2_INPUT_SIZE = 512
                img_resized_sam = resize_img(img_sam_np, SAM2_INPUT_SIZE)["unnormalized_img"]
                h_sam, w_sam = img_resized_sam.shape[:2]
                # Convert to CHW tensor [0, 1] for SAM2 state init
                img_tensor_sam = torch.from_numpy(img_resized_sam).permute(2, 0, 1).float().to(device) / 255.0
                img_tensor_sam = img_tensor_sam.unsqueeze(0) # 1xCxHxW
                # Prepare point prompts (Nx2 tensor) in (x,y) order
                points_sam = torch.tensor(point_prompts, dtype=torch.float, device=device).unsqueeze(0) # 1xNx2
                labels_sam = torch.ones(points_sam.shape[1], dtype=torch.int, device=device).unsqueeze(0) # 1xN (all foreground)
                with torch.no_grad():
                    sam2_refined_mask = None
                    state = sam2_predictor.init_state(video_path=img_tensor_sam)
                    sam2_predictor.add_new_points(state, frame_idx=0, obj_id=1, points=points_sam, labels=labels_sam)
                    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(state, start_frame_idx=0):
                        if out_frame_idx == 0 and 1 in out_obj_ids:
                            obj_idx = out_obj_ids.index(1)
                            sam2_refined_mask = (out_mask_logits[obj_idx] > 0.0)
                            if sam2_refined_mask.shape[0] == 1:
                                 sam2_refined_mask = sam2_refined_mask.squeeze(0)
                            break
                if sam2_refined_mask is not None:
                    # Resize SAM2 mask back to original frame dimensions (h, w)
                    sam2_refined_mask_np = sam2_refined_mask.cpu().numpy().astype(np.uint8)
                    sam2_refined_mask_resized_np = cv2.resize(sam2_refined_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    sam2_refined_mask_resized = torch.from_numpy(sam2_refined_mask_resized_np).to(device=device, dtype=torch.bool)
                    
                    if sam2_refined_mask_resized.shape != dynamic_mask.shape:
                        print(f"Warning: Resized SAM2 mask shape {sam2_refined_mask_resized.shape} mismatch original {dynamic_mask.shape} for frame {frame_i.frame_id}. Discarding SAM2 output.")
                    else:
                        original_sum = dynamic_mask.sum().item()
                        refined_sum = sam2_refined_mask_resized.sum().item()
                        dynamic_mask = sam2_refined_mask_resized
                        print(f"Dynamic mask for frame {frame_i.frame_id} refined with SAM2. Original sum: {original_sum}, Refined sum: {refined_sum}")
                        refined_successfully = True
            if not refined_successfully:
                 print(f"Warning: SAM2 refinement not applied for frame {frame_i.frame_id} (no points, SAM2 failed, or shape mismatch).")

        except Exception as e_sam:
            print(f"Error during SAM2 refinement for frame {frame_i.frame_id}: {e_sam}")

    return dynamic_mask, point_prompts 