import PIL
import numpy as np
import torch
import einops
from tqdm import tqdm
import lietorch
import os
# Add skimage import
from skimage.measure import label, regionprops
import cv2 # Add cv2 import for resizing

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import ImgNorm
from mast3r.model import AsymmetricMASt3R
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.config import config
import mast3r_slam.matching as matching
# TODO: check it uses the right dust3r
from thirdparty.monst3r.dust3r.inference import inference
from thirdparty.monst3r.dust3r.image_pairs import make_pairs
from thirdparty.monst3r.third_party.raft import load_RAFT
from thirdparty.monst3r.dust3r.utils.goem_opt import OccMask
from thirdparty.monst3r.dust3r.model import AsymmetricCroCo3DStereo
from thirdparty.monst3r.dust3r.utils.goem_opt import DepthBasedWarping
from thirdparty.monst3r.third_party.sam2.sam2.build_sam import build_sam2_video_predictor

_MONST3R_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_MONST3R_BASE_PATH = os.path.normpath(os.path.join(_MONST3R_UTILS_DIR, "..", "thirdparty", "monst3r"))
SAM2_CHECKPOINT_DEFAULT = os.path.join(_MONST3R_BASE_PATH, "third_party", "sam2", "checkpoints", "sam2.1_hiera_large.pt")
# Absolute path for existence check
SAM2_MODEL_CONFIG_ABSOLUTE_PATH = os.path.join(_MONST3R_BASE_PATH, "third_party", "sam2", "sam2", "configs", "sam2.1/sam2.1_hiera_l.yaml")
# Relative config name for Hydra, assuming build_sam.py initializes with config_path="configs" and expects the .yaml extension
SAM2_MODEL_CONFIG_NAME_FOR_HYDRA = "configs/sam2.1/sam2.1_hiera_l.yaml"

def load_mast3r(path=None, device="cuda"):
    weights_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if path is None
        else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model

def load_monst3r(path=None, device="cuda"):
    weights_path = (
        "thirdparty/monst3r/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
        if path is None
        else path
    )
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)
    return model


def load_retriever(mast3r_model, retriever_path=None, device="cuda"):
    retriever_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(retriever_path, backbone=mast3r_model, device=device)
    return retriever


@torch.inference_mode
def mast3r_decoder(mast3r, feat1, feat2, pos1, pos2, shape1, shape2):
    dec1, dec2 = mast3r._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = mast3r._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = mast3r._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2

@torch.inference_mode
def monst3r_decoder(monst3r, feat1, feat2, pos1, pos2, shape1, shape2):
    dec1, dec2 = monst3r._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = monst3r._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = monst3r._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def mast3r_downsample(X, C, D, Q):
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # C and Q: (...xHxW)
        # X and D: (...xHxWxF)
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q

def monst3r_downsample(X, C):
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # C and Q: (...xHxW)
        # X and D: (...xHxWxF)
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
    return X, C


@torch.inference_mode
def monst3r_symmetric_inference(mast3r, monst3r, frame_i, frame_j):
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = monst3r._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = monst3r._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    # MASt3R inference
    res11, res21 = mast3r_decoder(mast3r, feat1, feat2, pos1, pos2, shape1, shape2)
    res22, res12 = mast3r_decoder(mast3r, feat2, feat1, pos2, pos1, shape2, shape1)
    res = [res11, res21, res22, res12]
    _, _, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )

    # MonST3R inference
    res11, res21 = monst3r_decoder(monst3r, feat1, feat2, pos1, pos2, shape1, shape2)
    res22, res12 = monst3r_decoder(monst3r, feat2, feat1, pos2, pos1, shape2, shape1)
    res = [res11, res21, res22, res12]
    X, C = zip(
        *[(r["pts3d"][0], r["conf"][0]) for r in res]
    )

    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = mast3r_downsample(X, C, D, Q)
    return X, C, D, Q


# NOTE: Assumes img shape the same
@torch.inference_mode
def monst3r_decode_symmetric_batch(
    mast3r, monst3r, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
):
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]

        # MASt3R inference
        res11, res21 = mast3r_decoder(mast3r, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = mast3r_decoder(mast3r, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
        res = [res11, res21, res22, res12]
        _, _, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
        )

        # MonST3R inference
        res11, res21 = monst3r_decoder(monst3r, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = monst3r_decoder(monst3r, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
        res = [res11, res21, res22, res12]
        Xb, Cb = zip(
            *[(r["pts3d"][0], r["conf"][0]) for r in res]
        )

        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    X, C, D, Q = mast3r_downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode
def monst3r_inference_mono(monst3r, frame):
    """
    Mono inference using MonST3R - creates a self-pair for inference
    """
    if frame.feat is None:
        frame.feat, frame.pos, _ = monst3r._encode_image(frame.img, frame.img_true_shape)

    feat = frame.feat
    pos = frame.pos
    shape = frame.img_true_shape

    res11, res21 = monst3r_decoder(monst3r, feat, feat, pos, pos, shape, shape)
    res = [res11, res21]
    X, C = zip(
        *[(r["pts3d"][0], r["conf"][0]) for r in res]
    )
    # 2xhxwxc
    X, C = torch.stack(X), torch.stack(C)
    X, C = monst3r_downsample(X, C)

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    return Xii, Cii


def monst3r_match_symmetric(mast3r, monst3r, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    X, C, D, Q = monst3r_decode_symmetric_batch(
        mast3r, monst3r, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # Always matching both
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    # tic()
    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)
    # toc("Match")

    # TODO: Avoid this
    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )


@torch.inference_mode
def monst3r_asymmetric_inference(mast3r, monst3r, frame_i, frame_j):
    """
    Asymmetric inference using MonST3R - direct pairwise inference
    X, C obtained from MonST3R & D, Q obtained from MASt3R
    """

    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = monst3r._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = monst3r._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    # MonST3R inference
    res11, res21 = monst3r_decoder(monst3r, feat1, feat2, pos1, pos2, shape1, shape2)
    res = [res11, res21]
    X, C = zip(
        *[(r["pts3d"][0], r["conf"][0]) for r in res]
    )
    X, C = torch.stack(X), torch.stack(C)
    #X = torch.stack([pred1['pts3d'][0], pred2['pts3d_in_other_view'][0]], dim=0)
    #C = torch.stack([pred1['conf'][0], pred2['conf'][0]], dim=0)

    # Using MASt3R's encoded features since MONSt3R doesn't output features
    res11, res21 = mast3r_decoder(mast3r, feat1, feat2, pos1, pos2, shape1, shape2)
    res = [res11, res21]
    _, _, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # Create descriptors from encoded features from MASt3R
    D, Q = torch.stack(D), torch.stack(Q)
    
    # Ensure shape compatibility with existing code
    X, C, D, Q = mast3r_downsample(X, C, D, Q)
    
    return X, C, D, Q


def monst3r_match_asymmetric(mast3r, monst3r, frame_i, frame_j, idx_i2j_init=None):
    X, C, D, Q = monst3r_asymmetric_inference(mast3r=mast3r, 
                                              monst3r=monst3r, 
                                              frame_i=frame_i, 
                                              frame_j=frame_j)

    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Xii, Xji = X[:b], X[b:]
    Cii, Cji = C[:b], C[b:]
    Dii, Dji = D[:b], D[b:]
    Qii, Qji = Q[:b], Q[b:]

    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

    # How rest of system expects it
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji

@torch.inference_mode()
def get_dynamic_mask(monst3r, raft_model, frame_i, frame_j, threshold=0.35, refine_with_sam2=True, sam2_predictor=None):
    """
    Get dynamic mask between two frames using MonST3R and RAFT.
    Optionally refines the mask using SAM2.
    
    Compares optical flow (RAFT) with ego-motion flow (MonST3R depth + relative pose)
    to identify inconsistencies indicative of dynamic objects.
    
    Args:
        monst3r: Initialized MonST3R model.
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
        return empty_mask

    ## 2. Load RAFT model (Consider caching this outside if called frequently)
    #try:
    #    # Construct path relative to this file
    #    current_dir = os.path.dirname(os.path.abspath(__file__))
    #    raft_weights_path = os.path.join(current_dir, "../thirdparty/monst3r/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
    #    raft_weights_path = os.path.normpath(raft_weights_path) # Normalize path (e.g., remove ..)
    #    # Use the TartanAir checkpoint as in get_flow
    #    raft_model = load_RAFT(raft_weights_path)
    #    raft_model = raft_model.to(device)
    #    raft_model.eval()
    #except Exception as e:
    #    print(f"Error loading RAFT model: {e}")
    #    return empty_mask
    

    # 3. Compute Optical Flow (RAFT)
    try:
        # Prepare images for RAFT (needs BCHW format, scaled to [0, 255])
        # frame.img is already BCHW [-1, 1]
        img_i = frame_i.img # Should be Bx3xHxW
        img_j = frame_j.img # Should be Bx3xHxW

        # Ensure both tensors have the batch dimension (BCHW)
        if img_i.dim() == 3: img_i = img_i.unsqueeze(0)
        if img_j.dim() == 3: img_j = img_j.unsqueeze(0)
        
        # print(f"img_i.shape: {img_i.shape}") # Removed debug print
        # print(f"img_j.shape: {img_j.shape}") # Removed debug print

        # Rescale from [-1, 1] to [0, 255]
        img_i_raft = (img_i * 0.5 + 0.5) * 255.0
        img_j_raft = (img_j * 0.5 + 0.5) * 255.0

        flow_ij = raft_model(img_i_raft, img_j_raft, iters=20, test_mode=True)[1]
        # Output is Bx2xHxW
        flow_ij = flow_ij.squeeze(0) # Remove batch dim -> 2xHxW
    except Exception as e:
        print(f"Error computing optical flow: {e}")
        del raft_model # Clean up GPU memory
        return empty_mask
    finally:
         if 'raft_model' in locals() and raft_model is not None:
             del raft_model # Clean up GPU memory

    # 4. Get Relative Pose T_ji (transform points from i's frame to j's frame)
    T_WC_i = frame_i.T_WC
    T_WC_j = frame_j.T_WC
    if isinstance(T_WC_i, torch.Tensor): T_WC_i = lietorch.Sim3(T_WC_i)
    if isinstance(T_WC_j, torch.Tensor): T_WC_j = lietorch.Sim3(T_WC_j)

    # Assert that T_WC_i and T_WC_j are Sim3 objects after potential conversion
    assert isinstance(T_WC_i, lietorch.Sim3), f"T_WC_i is not a Sim3 object, but type {type(T_WC_i)}"
    assert isinstance(T_WC_j, lietorch.Sim3), f"T_WC_j is not a Sim3 object, but type {type(T_WC_j)}"

    T_ji = T_WC_j.inv() * T_WC_i

    # Extract rotation and translation (ensure correct types/shapes for DepthBasedWarping)
    # Extract from the 4x4 matrix
    T_ji_mat = T_ji.matrix()
    R_ji = T_ji_mat[..., :3, :3] # Extract Bx3x3 rotation
    T_ji_vec = T_ji_mat[..., :3, 3].unsqueeze(-1) # Extract Bx3 translation, add last dim

    # 5. Get Depth Map for frame_i
    try:
        if frame_i.feat is None:
             frame_i.feat, frame_i.pos, _ = monst3r._encode_image(frame_i.img, frame_i.img_true_shape)
        # Perform mono inference to get depth
        res_i, _ = monst3r_decoder(monst3r, frame_i.feat, frame_i.feat, frame_i.pos, frame_i.pos, frame_i.img_true_shape, frame_i.img_true_shape)
        depth_i = res_i['pts3d'][0, ..., 2] # HxW
        # DepthBasedWarping expects inverse depth (disparity) -> Bx1xHxW
        inv_depth_i = (1.0 / (depth_i + 1e-6)).unsqueeze(0).unsqueeze(0) # 1x1xHxW
    except Exception as e:
        print(f"Error computing depth map for frame_i: {e}")
        return empty_mask

    # 6. Get Intrinsics
    K_i = frame_i.K.unsqueeze(0) # 1x3x3
    K_j = frame_j.K.unsqueeze(0) # 1x3x3
    try:
        inv_K_i = torch.linalg.inv(K_i) # 1x3x3
    except Exception as e:
        print(f"Error inverting intrinsics K_i: {e}")
        return empty_mask


    # 7. Compute Ego-Motion Flow
    try:
        depth_warper = DepthBasedWarping().to(device)
        
        # ego_flow_ij: Bx3xHxW (flow_x, flow_y, mask)
        ego_flow_ij, _ = depth_warper(R_ji, T_ji_vec, R_ji.transpose(1,2), -R_ji.transpose(1,2) @ T_ji_vec, inv_depth_i, K_j, inv_K_i) # Use T_ij = inv(T_ji) ? Check docs -> expects pose1 R1,T1 and pose2 R2, T2
        # Let's re-check DepthBasedWarping call signature vs optimizer.py
        # optimizer.py: depth_wrapper(R_i, T_i, R_j, T_j, inv_depth_i, K_j, inv_K_i)
        # This warps points from frame i coordinates (using R_i, T_i world pose) to frame j coordinates (using R_j, T_j world pose)
        # We want to warp from frame i to frame j using the *relative* pose T_ji
        # Let P_i be a point in frame i. World point P_w = T_Wi * P_i
        # Project P_w into frame j: P_j = K_j * [I|0] * T_Wj.inv() * P_w
        # P_j = K_j * [I|0] * T_ji * P_i
        # The function seems to want world poses. Let's provide Identity for frame i and T_ji for frame j.
        R_i_world = torch.eye(3, device=device).unsqueeze(0) # 1x3x3
        T_i_world = torch.zeros((1, 3, 1), device=device)   # 1x3x1
        R_j_world = R_ji # 1x3x3
        T_j_world = T_ji # 1x3x1

        # ego_flow_ij, _ = depth_warper(R_i_world, T_i_world, R_j_world, T_j_world, inv_depth_i, K_j, inv_K_i)
        # Let's try the call signature from docs: DepthBasedWarping.__call__(self, R1, T1, R2, T2, disp1, K2, invK1):
        # R1, T1: pose of view 1; R2, T2: pose of view 2; disp1: disparity map of view 1; K2: intrinsics of view 2; invK1: inverse intrinsics of view 1
        # It calculates the flow FROM view 1 TO view 2.
        # So R1,T1 should be T_WC_i and R2,T2 should be T_WC_j
        # Directly use the matrix components as this was successful in fallback
        R1 = T_WC_i.matrix()[..., :3, :3]
        T1 = T_WC_i.matrix()[..., :3, 3].unsqueeze(-1)
        R2 = T_WC_j.matrix()[..., :3, :3]
        T2 = T_WC_j.matrix()[..., :3, 3].unsqueeze(-1)

        ego_flow_ij, _ = depth_warper(R1, T1, R2, T2, inv_depth_i, K_j, inv_K_i)

        ego_flow_ij = ego_flow_ij.squeeze(0) # Remove batch dim -> 3xHxW (flow_x, flow_y, mask)
    except AttributeError as e:
        print(f"Error extracting rotation/translation from T_WC_i or T_WC_j for ego-motion: {e}")
        #print(f"T_WC_i type: {type(T_WC_i)}, T_WC_j type: {type(T_WC_j)}")
        # Attempt fallback for world poses using .matrix()
        # try:
        #     print("Attempting fallback extraction for T_WC_i/T_WC_j using .matrix().")
        #     R1 = T_WC_i.matrix()[..., :3, :3]
        #     T1 = T_WC_i.matrix()[..., :3, 3].unsqueeze(-1)
        #     R2 = T_WC_j.matrix()[..., :3, :3]
        #     T2 = T_WC_j.matrix()[..., :3, 3].unsqueeze(-1)
        #     print("Fallback successful for T_WC_i/T_WC_j.")
        #     # Re-attempt ego-motion calculation with fallback poses
        #     ego_flow_ij, _ = depth_warper(R1, T1, R2, T2, inv_depth_i, K_j, inv_K_i)
        #     ego_flow_ij = ego_flow_ij.squeeze(0)
        # except Exception as fallback_e:
        #     print(f"Error during fallback extraction or ego-motion computation: {fallback_e}")
        #     if 'depth_warper' in locals(): del depth_warper
        #     return empty_mask
        # Since the direct .matrix() approach IS the fallback and seems to work, 
        # we'll keep it simple. If this primary method fails, the outer try-except will catch it.
        print(f"Primary Sim3 matrix extraction failed: {e}") # Should ideally not be reached if above is the fix
        if 'depth_warper' in locals() and hasattr(depth_warper, '__del__'): del depth_warper # Ensure cleanup
        return empty_mask

    except Exception as e:
        print(f"Error computing ego-motion flow: {e}")
        del depth_warper
        return empty_mask
    finally:
        if 'depth_warper' in locals() and depth_warper is not None:
            del depth_warper

    # 8. Compute Error Map
    # Use only the flow components (first 2 channels)
    flow_diff = flow_ij - ego_flow_ij[:2, ...]
    err_map = torch.norm(flow_diff, dim=0) # HxW

    # 9. Normalize and Threshold
    min_err = torch.min(err_map)
    max_err = torch.max(err_map)
    if max_err > min_err:
        norm_err_map = (err_map - min_err) / (max_err - min_err)
    else:
        norm_err_map = torch.zeros_like(err_map) # Avoid division by zero if error is constant

    dynamic_mask = norm_err_map > threshold # HxW boolean tensor, on device

    # 10. SAM2 Refinement (Optional) using logic from test_sam.py
    if refine_with_sam2 and sam2_predictor is not None and dynamic_mask.any():
        print(f"Attempting SAM2 refinement for frame {frame_i.frame_id}...")
        refined_successfully = False
        try:
            dynamic_mask_np = dynamic_mask.cpu().numpy()

            # Extract dynamic points using connected components
            labeled = label(dynamic_mask_np)
            regions = regionprops(labeled)
            point_prompts = []
            min_area = 20 # Ignore tiny regions
            for region in regions:
                if region.area >= min_area:
                    # centroid gives (row, col) i.e., (y, x)
                    y, x = region.centroid
                    point_prompts.append((int(x), int(y))) # Need (x,y) for SAM

            if len(point_prompts) > 0:
                # Prepare image for SAM2
                # frame_i.uimg is HxWx3, float [0,1]
                img_sam_np = frame_i.uimg.cpu().numpy()
                # Resize to SAM2 input size (e.g., 512 as in test_sam.py, assuming square)
                # Note: Using frame_i.uimg might be better quality than resizing frame_i.img
                # SAM2 often uses 1024, but let's stick to 512 from test_sam for now.
                # Check if resize_img expects HWC or CHW - it expects HWC np.uint8
                sam2_input_size = 512
                # resize_img returns uint8 HWC numpy array in 'unnormalized_img'
                img_resized_sam = resize_img(img_sam_np, sam2_input_size)["unnormalized_img"]
                h_sam, w_sam = img_resized_sam.shape[:2]

                # Convert to CHW tensor [0, 1] for SAM2 state init
                img_tensor_sam = torch.from_numpy(img_resized_sam).permute(2, 0, 1).float().to(device) / 255.0
                img_tensor_sam = img_tensor_sam.unsqueeze(0) # Add batch dim -> 1xCxHxW

                # Prepare point prompts (Nx2 tensor) in (x,y) order
                points_sam = torch.tensor(point_prompts, dtype=torch.float, device=device).unsqueeze(0) # 1xNx2
                labels_sam = torch.ones(points_sam.shape[1], dtype=torch.int, device=device).unsqueeze(0) # 1xN (all foreground)

                # Run SAM2 inference
                sam2_refined_mask = None
                state = sam2_predictor.init_state(video_path=img_tensor_sam)
                sam2_predictor.add_new_points(state, frame_idx=0, obj_id=1, points=points_sam, labels=labels_sam)

                for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(state, start_frame_idx=0):
                    if out_frame_idx == 0 and 1 in out_obj_ids:
                        obj_idx = out_obj_ids.index(1)
                        # Logits to boolean mask (HxW on device)
                        sam2_refined_mask = (out_mask_logits[obj_idx] > 0.0)
                        if sam2_refined_mask.shape[0] == 1: # Squeeze if batch dim is present
                             sam2_refined_mask = sam2_refined_mask.squeeze(0)
                        break # Only need the first frame's mask

                if sam2_refined_mask is not None:
                    # Resize SAM2 mask back to original frame dimensions (h, w)
                    sam2_refined_mask_np = sam2_refined_mask.cpu().numpy().astype(np.uint8)
                    sam2_refined_mask_resized_np = cv2.resize(sam2_refined_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    sam2_refined_mask_resized = torch.from_numpy(sam2_refined_mask_resized_np).to(device=device, dtype=torch.bool)

                    # Validate SAM2 output before combining
                    if sam2_refined_mask_resized.shape != dynamic_mask.shape:
                        print(f"Warning: Resized SAM2 mask shape {sam2_refined_mask_resized.shape} mismatch original {dynamic_mask.shape} for frame {frame_i.frame_id}. Discarding SAM2 output.")
                    else:
                        # Combine initial dynamic mask with SAM2 refinement using logical OR
                        original_sum = dynamic_mask.sum().item()
                        refined_sum = sam2_refined_mask_resized.sum().item()
                        dynamic_mask = dynamic_mask | sam2_refined_mask_resized
                        combined_sum = dynamic_mask.sum().item()
                        print(f"Dynamic mask for frame {frame_i.frame_id} refined with SAM2. Original sum: {original_sum}, Refined sum: {refined_sum}, Combined sum: {combined_sum}")
                        refined_successfully = True

            if not refined_successfully:
                 print(f"Warning: SAM2 refinement not applied for frame {frame_i.frame_id} (no points, SAM2 failed, or shape mismatch).")

        except Exception as e_sam:
            print(f"Error during SAM2 refinement for frame {frame_i.frame_id}: {e_sam}")
        # Removed old SAM2 block

    # TODO: Implement SAM2 refinement using the new technique in test_sam.py - DONE

    return dynamic_mask # on original device

    
def get_flow(self, sintel_ckpt=False): #TODO: test with gt flow
    print('precomputing flow...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    get_valid_flow_mask = OccMask(th=3.0)
    pair_imgs = [np.stack(self.imgs)[self._ei], np.stack(self.imgs)[self._ej]]
    flow_net = load_RAFT() if sintel_ckpt else load_RAFT("thirdparty/monst3r/dust3r/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
    flow_net = flow_net.to(device)
    flow_net.eval()
    with torch.no_grad():
        chunk_size = 12
        flow_ij = []
        flow_ji = []
        num_pairs = len(pair_imgs[0])
        for i in tqdm(range(0, num_pairs, chunk_size)):
            end_idx = min(i + chunk_size, num_pairs)
            imgs_ij = [torch.tensor(pair_imgs[0][i:end_idx]).float().to(device),
                    torch.tensor(pair_imgs[1][i:end_idx]).float().to(device)]
            flow_ij.append(flow_net(imgs_ij[0].permute(0, 3, 1, 2) * 255, 
                                    imgs_ij[1].permute(0, 3, 1, 2) * 255, 
                                    iters=20, test_mode=True)[1])
            flow_ji.append(flow_net(imgs_ij[1].permute(0, 3, 1, 2) * 255, 
                                    imgs_ij[0].permute(0, 3, 1, 2) * 255, 
                                    iters=20, test_mode=True)[1])
        flow_ij = torch.cat(flow_ij, dim=0)
        flow_ji = torch.cat(flow_ji, dim=0)
        valid_mask_i = get_valid_flow_mask(flow_ij, flow_ji)
        valid_mask_j = get_valid_flow_mask(flow_ji, flow_ij)
    print('flow precomputed')
    # delete the flow net
    if flow_net is not None: del flow_net
    return flow_ij, flow_ji, valid_mask_i, valid_mask_j


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(img, size, square_ok=False, return_transformation=False):
    assert size == 224 or size == 512
    # numpy to PIL format
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    res = dict(
        img=ImgNorm(img)[None],
        true_shape=np.int32([img.size[::-1]]),
        unnormalized_img=np.asarray(img),
    )
    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res
