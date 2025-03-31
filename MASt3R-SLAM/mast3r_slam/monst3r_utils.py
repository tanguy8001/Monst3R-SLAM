import PIL
import numpy as np
import torch
import einops

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import ImgNorm
from dust3r.model import AsymmetricCroCo3DStereo
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.config import config
import mast3r_slam.matching as matching
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs


def load_monst3r(path=None, device="cuda"):
    weights_path = (
        "/work/courses/3dv/24/monst3r/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
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
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def downsample(X, C, D, Q):
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # C and Q: (...xHxW)
        # X and D: (...xHxWxF)
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q


@torch.inference_mode
def monst3r_symmetric_inference(model, frame_i, frame_j):
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1)
    res = [res11, res21, res22, res12]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


# NOTE: Assumes img shape the same
@torch.inference_mode
def monst3r_decode_symmetric_batch(
    model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
):
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
        res = [res11, res21, res22, res12]
        Xb, Cb, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
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
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode
def monst3r_inference_mono(model, frame):
    """
    Mono inference using MonST3R - creates a self-pair for inference
    """
    if frame.feat is None:
        frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)

    # Create a pair with itself for MonST3R
    img_dict = {
        'img': frame.img,
        'idx': 0,
        'true_shape': frame.img_true_shape
    }
    
    # Create a pair with itself
    pairs = make_pairs([img_dict, img_dict.copy()], scene_graph='complete')
    
    # Run MonST3R inference
    with torch.no_grad():
        output = inference(pairs, model, frame.img.device, batch_size=1, verbose=False)
    
    # Get pointcloud and confidence from the first prediction
    pts3d = output['pred1']['pts3d'][0]  # [H, W, 3]
    conf = output['pred1']['conf'][0]    # [H, W]
    
    # Reshape for SLAM system
    Xii = pts3d.reshape(1, -1, 3)
    Cii = conf.reshape(1, -1, 1)

    return Xii, Cii


def monst3r_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    X, C, D, Q = monst3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
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
def monst3r_asymmetric_inference(model, frame_i, frame_j):
    """
    Asymmetric inference using MonST3R - direct pairwise inference
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
    
    # Create a pair with asymmetric structure (i->j)
    pairs = [(img_i, img_j)]
    
    # Run MonST3R inference
    with torch.no_grad():
        output = inference(pairs, model, frame_i.img.device, batch_size=1, verbose=False)
    
    # Extract predictions
    pred1, pred2 = output['pred1'], output['pred2']
    
    # Get 3D points and descriptors
    X = torch.stack([pred1['pts3d'][0], pred2['pts3d_in_other_view'][0]], dim=0)
    C = torch.stack([pred1['conf'][0], pred2['conf'][0]], dim=0)
    
    # For compatibility with existing code, create descriptor tensors
    # Use the encoded features as descriptors if available
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )
    
    # Using MASt3R's encoded features since MONSt3R doesn't output features
    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res = [res11, res21]
    _, _, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    # Create descriptors from encoded features from MASt3R
    D, Q = torch.stack(D), torch.stack(Q)
    
    # Ensure shape compatibility with existing code
    X, C, D, Q = downsample(X, C, D, Q)
    
    return X, C, D, Q


def monst3r_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    X, C, D, Q = monst3r_asymmetric_inference(model, frame_i, frame_j)

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


def get_dynamic_mask(model, frame_i, frame_j, threshold=0.35):
    """
    Get dynamic mask between two frames using MonST3R
    Returns a binary mask where True indicates dynamic content
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
    
    # Create pairs in both directions
    pairs = [(img_i, img_j), (img_j, img_i)]
    
    # Run MonST3R inference
    with torch.no_grad():
        output = inference(pairs, model, frame_i.img.device, batch_size=1, verbose=False)
    
    # Get predictions
    pred1_forward = output['pred1'][0]  # i->j
    pred2_forward = output['pred2'][0]  # i->j result
    pred1_backward = output['pred1'][1]  # j->i
    pred2_backward = output['pred2'][1]  # j->i result
    
    # Compute reprojection error between the two directions
    # This is a simplified method - actual implementation would need more details
    # from get_dynamic_mask_from_pairviewer in demo.py
    pts3d_i = pred1_forward['pts3d']
    pts3d_j_in_i = pred2_forward['pts3d_in_other_view']
    
    # Calculate reprojection error
    error_map = torch.norm(pts3d_i - pts3d_j_in_i, dim=-1)
    
    # Normalize error map
    error_min = error_map.min()
    error_max = error_map.max()
    normalized_error = (error_map - error_min) / (error_max - error_min + 1e-8)
    
    # Binary threshold
    dynamic_mask = (normalized_error > threshold).float()
    
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
