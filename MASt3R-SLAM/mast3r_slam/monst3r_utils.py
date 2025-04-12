import PIL
import numpy as np
import torch
import einops
from tqdm import tqdm
import lietorch
import os

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
def get_dynamic_mask(monst3r, frame_i, frame_j, threshold=0.35):
    """
    Get dynamic mask between two frames using MonST3R and RAFT.
    
    Compares optical flow (RAFT) with ego-motion flow (MonST3R depth + relative pose)
    to identify inconsistencies indicative of dynamic objects.
    
    Args:
        monst3r: Initialized MonST3R model.
        frame_i: First frame object (needs img, T_WC, K).
        frame_j: Second frame object (needs img, T_WC, K).
        threshold: Threshold for classifying pixels as dynamic based on normalized flow error (0.0-1.0).
        
    Returns:
        dynamic_mask: Binary mask (HxW tensor) where True indicates dynamic content.
                      Returns an empty mask if calculation fails (e.g., missing calibration).
    """
    device = frame_i.img.device
    h, w = frame_i.img.shape[-2:]
    empty_mask = torch.zeros((h, w), dtype=torch.bool, device=device)

    # 1. Check for calibration (required for ego-motion flow)
    if not hasattr(frame_i, 'K') or not hasattr(frame_j, 'K') or frame_i.K is None or frame_j.K is None:
        print("Warning: Cannot compute dynamic mask without camera calibration (K).")
        return empty_mask

    # 2. Load RAFT model (Consider caching this outside if called frequently)
    try:
        # Construct path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        raft_weights_path = os.path.join(current_dir, "../thirdparty/monst3r/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
        raft_weights_path = os.path.normpath(raft_weights_path) # Normalize path (e.g., remove ..)

        # Use the TartanAir checkpoint as in get_flow
        raft_model = load_RAFT(raft_weights_path)
        raft_model = raft_model.to(device)
        raft_model.eval()
    except Exception as e:
        print(f"Error loading RAFT model: {e}")
        return empty_mask

    # 3. Compute Optical Flow (RAFT)
    try:
        # Prepare images for RAFT (needs BGR, 0-255, CHW format)
        img_i_raft = frame_i.img.permute(0, 3, 1, 2) * 255
        img_j_raft = frame_j.img.permute(0, 3, 1, 2) * 255
        # Ensure batch dimension is present
        if img_i_raft.dim() == 3: img_i_raft = img_i_raft.unsqueeze(0)
        if img_j_raft.dim() == 3: img_j_raft = img_j_raft.unsqueeze(0)

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

    # T_ji = T_jW * T_Wi = T_WC_j.inv() * T_WC_i # Incorrect: This maps world points in frame i coord system to world points in frame j coord system
    # Correct: T_ji transforms points FROM frame i coordinate system TO frame j coordinate system
    # Point_j = T_ji * Point_i
    # Point_w = T_Wi * Point_i => Point_i = T_Wi.inv() * Point_w
    # Point_w = T_Wj * Point_j => Point_j = T_Wj.inv() * Point_w
    # Point_j = T_Wj.inv() * T_Wi * Point_i
    T_ji = T_WC_j.inv() * T_WC_i

    # Extract rotation and translation (ensure correct types/shapes for DepthBasedWarping)
    # DepthBasedWarping expects R (Bx3x3), T (Bx3x1)
    R_ji = T_ji.rotation().matrix().unsqueeze(0)  # 1x3x3
    T_ji = T_ji.translation().unsqueeze(-1).unsqueeze(0) # 1x3x1

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
        ego_flow_ij, _ = depth_warper(R_ji, T_ji, R_ji.transpose(1,2), -R_ji.transpose(1,2) @ T_ji, inv_depth_i, K_j, inv_K_i) # Use T_ij = inv(T_ji) ? Check docs -> expects pose1 R1,T1 and pose2 R2, T2
        # Let's re-check DepthBasedWarping call signature vs optimizer.py
        # optimizer.py: depth_wrapper(R_i, T_i, R_j, T_j, inv_depth_i, K_j, inv_K_i)
        # This warps points from frame i coordinates (using R_i, T_i world pose) to frame j coordinates (using R_j, T_j world pose)
        # We want to warp from frame i to frame j using the *relative* pose T_ji
        # Let P_i be a point in frame i. World point P_w = T_Wi * P_i
        # Project P_w into frame j: P_j = K_j * [I|0] * T_Wj.inv() * P_w
        # P_j = K_j * [I|0] * T_Wj.inv() * T_Wi * P_i
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
        R1 = T_WC_i.rotation().matrix().unsqueeze(0)
        T1 = T_WC_i.translation().unsqueeze(-1).unsqueeze(0)
        R2 = T_WC_j.rotation().matrix().unsqueeze(0)
        T2 = T_WC_j.translation().unsqueeze(-1).unsqueeze(0)

        ego_flow_ij, _ = depth_warper(R1, T1, R2, T2, inv_depth_i, K_j, inv_K_i)


        ego_flow_ij = ego_flow_ij.squeeze(0) # Remove batch dim -> 3xHxW (flow_x, flow_y, mask)
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

    dynamic_mask = norm_err_map > threshold # HxW boolean tensor

    return dynamic_mask

    
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
