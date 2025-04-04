import PIL
import numpy as np
import torch
import einops

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import ImgNorm
from mast3r.model import AsymmetricMASt3R
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.config import config
import mast3r_slam.matching as matching
# TODO: check it uses the right dust3r
from thirdparty.monst3r.dust3r.inference import inference
from thirdparty.monst3r.dust3r.image_pairs import make_pairs
from thirdparty.monst3r.dust3r.model import AsymmetricCroCo3DStereo

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

def get_dynamic_mask(monst3r, frame_i, frame_j, threshold=0.35):
    """
    Get dynamic mask between two frames using MonST3R
    
    This function uses the pairwise pointmaps from MonST3R to identify dynamic objects
    by comparing the stereo reconstruction from two viewpoints. Areas with high reprojection
    error/geometric inconsistency are likely to be dynamic objects.
    
    Args:
        monst3r: MonST3R model
        frame_i: First frame
        frame_j: Second frame
        threshold: Threshold for classifying pixels as dynamic (0.0-1.0)
        
    Returns:
        dynamic_mask: Binary mask where 1 indicates dynamic content (H×W tensor)
    """
    # Create image dictionaries for MonST3R
    img_i = {
        'img': frame_i.img,
        'idx': 0,
        'true_shape': frame_i.img_true_shape
    }
    
    img_j = {
        'img': frame_j.img,
        'idx': 1,
        'true_shape': frame_j.img_true_shape
    }
    
    # Create pairs in both directions for better dynamic detection
    pairs = make_pairs([img_i, img_j])
    
    # Run MonST3R inference to get pointmaps
    with torch.no_grad():
        outputs = inference(pairs, monst3r, device=frame_i.img.device, batch_size=1, verbose=False)
    
    # Get the pointmaps from both directions
    # i->j direction
    forward_pts3d = outputs[0]['pred1']['pts3d'][0]  # (H, W, 3)
    forward_pts3d_in_other = outputs[0]['pred2']['pts3d_in_other_view'][0]  # (H, W, 3)
    
    # j->i direction
    backward_pts3d = outputs[1]['pred1']['pts3d'][0]  # (H, W, 3)
    backward_pts3d_in_other = outputs[1]['pred2']['pts3d_in_other_view'][0]  # (H, W, 3)
    
    # Get the confidence maps
    forward_conf = outputs[0]['pred1']['conf'][0]  # (H, W)
    backward_conf = outputs[1]['pred1']['conf'][0]  # (H, W)
    
    # Compute geometric inconsistency in forward direction
    # For each point in frame i, we check if its reconstruction is consistent when viewed from frame j
    forward_error = torch.norm(forward_pts3d - forward_pts3d_in_other, dim=-1)  # (H, W)
    
    # Similarly for backward direction
    backward_error = torch.norm(backward_pts3d - backward_pts3d_in_other, dim=-1)  # (H, W)
    
    # Combine errors from both directions, weighted by confidence
    combined_error = (forward_error * forward_conf + backward_error * backward_conf) / (forward_conf + backward_conf + 1e-8)
    
    # Normalize error map to range [0, 1]
    error_min = combined_error.min()
    error_max = combined_error.max()
    normalized_error = (combined_error - error_min) / (error_max - error_min + 1e-8)
    
    # Apply threshold to get binary mask
    dynamic_mask = (normalized_error > threshold)
    
    # Optional: Apply morphological operations to clean up the mask
    # Convert to numpy for cv2 operations
    if dynamic_mask.numel() > 0:
        import cv2
        import numpy as np
        
        mask_np = dynamic_mask.cpu().numpy().astype(np.uint8)
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to tensor
        dynamic_mask = torch.from_numpy(mask_np).to(dynamic_mask.device)
    
    return dynamic_mask


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
