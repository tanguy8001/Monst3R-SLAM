import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import lietorch
from skimage.measure import label, regionprops

from mast3r_slam.dataloader import load_dataset
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.monst3r_utils import (
    SAM2_CHECKPOINT_DEFAULT,
    SAM2_MODEL_CONFIG_NAME_FOR_HYDRA,
    build_sam2_video_predictor,
    resize_img,
    get_dynamic_mask,
    load_monst3r
)
from mast3r_slam.frame import create_frame
from thirdparty.monst3r.third_party.raft import load_RAFT

def save_mask_overlay(img, mask, save_path):
    """
    Save an image with a semi-transparent red overlay where mask is True.
    img: numpy array, HxWx3, float [0,1] or uint8
    mask: numpy array, HxW, bool or 0/1
    save_path: str
    """
    mask = np.squeeze(mask) # Ensure mask is 2D
    if mask.ndim != 2:
        print(f"Warning: Mask shape after squeeze is still {mask.shape}, expected 2D. Skipping overlay.")
        mask = np.zeros_like(img[:,:,0], dtype=np.uint8)

    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    img_pil = Image.fromarray(img).convert("RGBA")
    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    mask_rgba[mask > 0] = (255, 0, 0, 128)
    mask_img = Image.fromarray(mask_rgba, 'RGBA')
    overlay.paste(mask_img, (0, 0), mask_img)
    out = Image.alpha_composite(img_pil, overlay).convert("RGB")
    out.save(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to video file or dataset directory')
    parser.add_argument('--num-frames', type=int, default=10, help='Number of frames to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device for models (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='logs/sam2_test', help='Output directory')
    parser.add_argument('--config', default='config/calib.yaml')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    load_config(args.config)
    print(config)
    dataset = load_dataset(args.video)
    dataset.subsample(config["dataset"]["subsample"])
    print(f"Loaded dataset with {len(dataset)} frames.")

    device = args.device
    monst3r = load_monst3r(device=device)
    raft_model = load_RAFT("thirdparty/monst3r/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
    raft_model = raft_model.to(device)
    raft_model.eval()
    sam2_predictor = build_sam2_video_predictor(
        SAM2_MODEL_CONFIG_NAME_FOR_HYDRA,
        SAM2_CHECKPOINT_DEFAULT,
        device=device
    )
    sam2_predictor.eval()

    K = None
    if hasattr(dataset, 'camera_intrinsics') and dataset.camera_intrinsics is not None:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(device, dtype=torch.float32)

    for idx in tqdm(range(1, min(args.num_frames, len(dataset)))):
        _, img_prev = dataset[idx-1]
        _, img = dataset[idx]
        T_WC = lietorch.Sim3.Identity(1, device=device)
        frame_prev = create_frame(idx-1, img_prev, T_WC, K=K, img_size=dataset.img_size, device=device)
        frame = create_frame(idx, img, T_WC, K=K, img_size=dataset.img_size, device=device)

        # --- Step A: Compute dynamic mask (RAFT + ego-motion) ---
        dynamic_mask = get_dynamic_mask(monst3r, raft_model, frame, frame_prev, threshold=0.35, refine_with_sam2=False)
        dynamic_mask_np = dynamic_mask.cpu().numpy().astype(np.uint8)

        print(f"Dynamic mask stats for frame {idx}: unique={np.unique(dynamic_mask_np)}, sum={np.sum(dynamic_mask_np)}")

        # --- Step B: Extract dynamic points (connected components) ---
        labeled = label(dynamic_mask_np)
        regions = regionprops(labeled)
        point_prompts = []
        min_area = 20  # Ignore tiny regions
        for region in regions:
            if region.area > min_area:
                y, x = region.centroid
                point_prompts.append((int(x), int(y)))

        # --- Step C: Use dynamic points as prompts for SAM2 ---
        # Point prompt generation based on centroids of connected components
        # Prepare image for SAM2 (resize to 512x512, uint8)
        img_resized = resize_img(img, 512)["unnormalized_img"]
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0  # CHW
        img_tensor = img_tensor.unsqueeze(0).to(device)
        # Prepare point prompts for SAM2 (N,2) in (x,y) order
        if len(point_prompts) > 0:
            points = torch.tensor(point_prompts, dtype=torch.float, device=device).unsqueeze(0)  # 1xNx2
            labels = torch.ones(points.shape[1], dtype=torch.int, device=device).unsqueeze(0)    # 1xN, all foreground

            with torch.no_grad():
                state = sam2_predictor.init_state(video_path=img_tensor)
                sam2_predictor.add_new_points(state, frame_idx=0, obj_id=1, points=points, labels=labels)
                mask = None

                for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(state, start_frame_idx=0):
                    
                    if out_frame_idx == 0 and 1 in out_obj_ids:
                        obj_idx = out_obj_ids.index(1)
                        mask = (out_mask_logits[obj_idx] > 0).cpu().numpy().astype(np.uint8)
                        

                if mask is None:
                    mask = np.zeros(img_resized.shape[:2], dtype=np.uint8)
        else:
            mask = np.zeros(img_resized.shape[:2], dtype=np.uint8)

        #print(f"Mask stats for frame {idx}: shape={mask.shape}, unique={np.unique(mask)}, sum={np.sum(mask)}")

        save_path = os.path.join(args.output_dir, f"frame_{idx:04d}.png")
        #print(f"Saving mask to {save_path}")
        save_mask_overlay(img_resized, mask, save_path)
        
        #mask_path = os.path.join(args.output_dir, f"frame_{idx:04d}_mask.png")
        #mask_to_save = np.squeeze(mask)
        #Image.fromarray((mask_to_save * 255).astype(np.uint8)).save(mask_path)

if __name__ == "__main__":
    main()
