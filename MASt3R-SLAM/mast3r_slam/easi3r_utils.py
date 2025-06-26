import PIL
import numpy as np
import torch
import einops
import os
import cv2

from thirdparty.Easi3R.dust3r.utils.image import ImgNorm
from thirdparty.Easi3R.dust3r.model import AsymmetricCroCo3DStereo
from thirdparty.Easi3R.dust3r.inference import inference
from thirdparty.Easi3R.dust3r.image_pairs import make_pairs
from thirdparty.Easi3R.dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from thirdparty.mast3r.mast3r.model import AsymmetricMASt3R
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.config import config
import mast3r_slam.matching as matching


def load_easi3r(path=None, device="cuda"):
    """Load Easi3R model (based on DUSt3R architecture)"""
    weights_path = (
        "thirdparty/Easi3R/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        if path is None
        else path
    )
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)
    return model


def load_mast3r(path=None, device="cuda"):
    """Load MASt3R model for feature extraction"""
    weights_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if path is None
        else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model


def load_retriever(mast3r_model, retriever_path=None, device="cuda"):
    """Load retrieval database using MASt3R backbone"""
    retriever_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(retriever_path, backbone=mast3r_model, device=device)
    return retriever


def easi3r_double_inference_pair(easi3r_model, frame_i, frame_j):
    """
    Perform Easi3R double inference for a pair of frames
    Returns pointmaps with dynamic objects removed
    """
    # Prepare images for inference
    imgs = []
    
    # Convert frames to the format expected by Easi3R
    for idx, frame in enumerate([frame_i, frame_j]):
        if hasattr(frame, 'unnormalized_img'):
            img_array = frame.unnormalized_img
        else:
            # Convert from normalized image
            img_array = (frame.img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        img_dict = {
            'img': frame.img,
            'true_shape': frame.img_true_shape,
            'idx': idx,
            'instance': str(idx)
        }
        imgs.append(img_dict)
    
    # First inference - extract dynamic masks
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    preds_first = inference(pairs, easi3r_model, frame_i.img.device, batch_size=1, verbose=False)
    
    # Global alignment without flow_loss_weight parameter to extract dynamic masks
    scene_dynamic = global_aligner(
        preds_first, 
        device=frame_i.img.device, 
        mode=GlobalAlignerMode.PointCloudOptimizer,
        verbose=False
    )
    
    # Extract dynamic masks
    dynamic_masks = getattr(scene_dynamic, 'dynamic_masks', None)
    
    # Clean up first inference
    del preds_first, scene_dynamic
    torch.cuda.empty_cache()
    
    # Second inference - apply attention reweighting
    if dynamic_masks is not None:
        # Attach masks to images
        for i, img_dict in enumerate(imgs):
            if i < len(dynamic_masks):
                img_dict['atten_mask'] = dynamic_masks[i].cpu().unsqueeze(0)
    
    # Create new pairs with masks attached
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    preds_second = inference(pairs, easi3r_model, frame_i.img.device, batch_size=1, verbose=False)
    
    # Final global alignment with attention reweighting
    scene = global_aligner(
        preds_second,
        device=frame_i.img.device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        use_atten_mask=True,  # Enable attention reweighting
        verbose=False
    )
    
    # Extract filtered pointmaps
    pts3d = scene.get_pts3d(raw_pts=True)
    confs = [c for c in scene.im_conf]
    
    # Convert to format expected by SLAM system
    X_i = pts3d[0]  # Points for frame i
    X_j = pts3d[1]  # Points for frame j
    C_i = confs[0]  # Confidence for frame i
    C_j = confs[1]  # Confidence for frame j

    return X_i, C_i, X_j, C_j


# Process features through the decoder (reused from mast3r_utils)
@torch.inference_mode
def decoder(mast3r_model, feat1, feat2, pos1, pos2, shape1, shape2):
    dec1, dec2 = mast3r_model._decoder(feat1, pos1, feat2, pos2, shape1, shape2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = mast3r_model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = mast3r_model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def downsample(X, C, D, Q):
    """Downsample pointmaps and features"""
    downsample_factor = config["dataset"]["img_downsample"]
    if downsample_factor > 1:
        X = X[..., ::downsample_factor, ::downsample_factor, :].contiguous()
        C = C[..., ::downsample_factor, ::downsample_factor].contiguous()
        D = D[..., ::downsample_factor, ::downsample_factor, :].contiguous()
        Q = Q[..., ::downsample_factor, ::downsample_factor].contiguous()
    return X, C, D, Q


@torch.inference_mode
def easi3r_inference_mono(easi3r_model, mast3r_model, frame):
    """
    Mono inference combining Easi3R filtered pointmaps with MASt3R descriptors
    """
    # Get filtered pointmaps from Easi3R
    # Prepare single image for inference
    img_dict = {
        'img': frame.img,
        'true_shape': frame.img_true_shape,
        'idx': 0,
        'instance': '0'
    }
    
    # For mono inference, we duplicate the image
    imgs = [img_dict, img_dict.copy()]
    imgs[1]['idx'] = 1
    imgs[1]['instance'] = '1'
    
    # First inference - extract dynamic masks
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    preds_first = inference(pairs, easi3r_model, frame.img.device, batch_size=1, verbose=False)
    
    # Global alignment without flow_loss_weight parameter
    scene_dynamic = global_aligner(
        preds_first,
        device=frame.img.device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        verbose=False
    )
    
    # Extract dynamic masks
    dynamic_masks = getattr(scene_dynamic, 'dynamic_masks', None)
    
    # Clean up first inference
    del preds_first, scene_dynamic
    torch.cuda.empty_cache()
    
    # Second inference with attention reweighting
    if dynamic_masks is not None:
        for i, img_dict in enumerate(imgs):
            if i < len(dynamic_masks):
                img_dict['atten_mask'] = dynamic_masks[i].cpu().unsqueeze(0)
    
    # Create new pairs with masks
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    preds_second = inference(pairs, easi3r_model, frame.img.device, batch_size=1, verbose=False)
    
    # Final alignment with reweighting
    scene = global_aligner(
        preds_second,
        device=frame.img.device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        use_atten_mask=True,
        verbose=False
    )
    
    # Extract pointmap for the frame
    pts3d = scene.get_pts3d(raw_pts=True)[0]  # Take first (same as second for mono)
    conf = scene.im_conf[0]

    # Convert to format expected by SLAM system
    # Reshape to (N, 3) and (N, 1) format
    X = einops.rearrange(pts3d, "h w c -> (h w) c")
    C = einops.rearrange(conf, "h w -> (h w) 1")
    
    return X, C


@torch.inference_mode
def easi3r_match_asymmetric(easi3r_model, mast3r_model, frame_i, frame_j, idx_i2j_init=None):
    """
    Asymmetric matching with Easi3R filtered pointmaps and MASt3R descriptors
    """
    # Get filtered pointmaps from Easi3R double inference
    X_i_filtered, C_i_filtered, X_j_filtered, C_j_filtered = easi3r_double_inference_pair(easi3r_model, frame_i, frame_j)
    
    # Get MASt3R features and descriptors
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = mast3r_model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = mast3r_model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    # Get MASt3R descriptors (asymmetric inference)
    res11, res21 = decoder(mast3r_model, feat1, feat2, pos1, pos2, shape1, shape2)
    res = [res11, res21]
    
    # Extract descriptors and descriptor confidences from MASt3R
    D, Q = zip(*[(r["desc"][0], r["desc_conf"][0]) for r in res])
    D, Q = torch.stack(D), torch.stack(Q)
    
    # Stack the filtered pointmaps from Easi3R
    X = torch.stack([X_i_filtered, X_j_filtered])
    C = torch.stack([C_i_filtered, C_j_filtered])
    
    # Apply downsampling consistently
    X, C, D, Q = downsample(X, C, D, Q)

    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2 if b > 1 else 1

    Xii, Xji = X[:b], X[b:] if b < len(X) else X[:b]
    Cii, Cji = C[:b], C[b:] if b < len(C) else C[:b]  
    Dii, Dji = D[:b], D[b:] if b < len(D) else D[:b]
    Qii, Qji = Q[:b], Q[b:] if b < len(Q) else Q[:b]

    # Perform matching using MASt3R descriptors but Easi3R filtered pointmaps
    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

    # Reshape to expected format
    Xii = einops.rearrange(Xii, "b h w c -> b (h w) c")
    Cii = einops.rearrange(Cii, "b h w -> b (h w) 1")
    Dii = einops.rearrange(Dii, "b h w c -> b (h w) c")
    Qii = einops.rearrange(Qii, "b h w -> b (h w) 1")
    
    Xji = einops.rearrange(Xji, "b h w c -> b (h w) c")
    Cji = einops.rearrange(Cji, "b h w -> b (h w) 1")
    Dji = einops.rearrange(Dji, "b h w c -> b (h w) c")
    Qji = einops.rearrange(Qji, "b h w -> b (h w) 1")

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji


def _resize_pil_image(img, long_edge_size):
    """Resize PIL image maintaining aspect ratio"""
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(img, size, square_ok=False, return_transformation=False):
    """Resize image for model input"""
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