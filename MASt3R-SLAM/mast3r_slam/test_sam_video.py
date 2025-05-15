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
    
    # Ensure overlay is the same size as img_pil
    if mask_img.size != img_pil.size:
        print(f"Warning: Mask image size {mask_img.size} differs from base image size {img_pil.size}. Resizing mask overlay to fit.")
        mask_img = mask_img.resize(img_pil.size, Image.NEAREST)

    overlay.paste(mask_img, (0, 0), mask_img)
    out = Image.alpha_composite(img_pil, overlay).convert("RGB")
    out.save(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to video file or dataset directory')
    parser.add_argument('--num-frames', type=int, default=10, help='Number of frames to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device for models (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='logs/sam2_test_video', help='Output directory')
    parser.add_argument('--config', default='config/calib.yaml')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    load_config(args.config)
    print(config)
    dataset = load_dataset(args.video)
    dataset.subsample(config["dataset"]["subsample"])
    print(f"Loaded dataset with {len(dataset)} frames.")

    if len(dataset) < 2:
        print(f"Error: Dataset must contain at least 2 frames after subsampling to proceed for dynamic mask computation. Found {len(dataset)} frames.")
        print("Please check the video path, image content in the directory, or the subsampling rate in the config.")
        return

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

    video_dir = 'logs/sam2_test_frames'
    frame_names = [f for f in os.listdir(video_dir) if f.lower().endswith('.jpg')]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    vis_frame_stride = 1
    
    idx = 1
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
    if len(point_prompts) > 0:
        #points = torch.tensor(point_prompts, dtype=torch.float, device=device).unsqueeze(0)  # 1xNx2
        points = torch.tensor(np.array([[190, 480]]), dtype=torch.float, device=device).unsqueeze(0)  # 1xNx2
        labels = torch.ones(points.shape[1], dtype=torch.int, device=device).unsqueeze(0)    # 1xN, all foreground
        with torch.no_grad():
            state = sam2_predictor.init_state(video_path='logs/sam2_test_frames')
            sam2_predictor.add_new_points(state, frame_idx=0, obj_id=1, points=points, labels=labels)
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                if out_frame_idx > args.num_frames:
                    break
    else:
        video_segments = {}

    if not video_segments:
        print("No video segments were generated by SAM. No overlay images will be saved.")
    else:
        processed_sam_frame_indices = sorted(video_segments.keys())

        for sam_frame_idx in tqdm(processed_sam_frame_indices, desc="Saving frames with masks"):
            if sam_frame_idx >= len(frame_names):
                print(f"SAM output frame index {sam_frame_idx} is out of bounds for available image files (total: {len(frame_names)}). Skipping.")
                continue
            
            if sam_frame_idx > args.num_frames: 
                 print(f"Skipping SAM frame index {sam_frame_idx} as it exceeds args.num_frames ({args.num_frames}).")
                 continue

            current_frame_filename = frame_names[sam_frame_idx]
            image_path = os.path.join(video_dir, current_frame_filename)

            try:
                img_pil = Image.open(image_path).convert("RGB")
                img_np = np.array(img_pil)
            except FileNotFoundError:
                print(f"Image file not found: {image_path}. Skipping frame {current_frame_filename}.")
                continue
            except Exception as e:
                print(f"Error loading image {image_path}: {e}. Skipping frame {current_frame_filename}.")
                continue

            frame_specific_masks = video_segments.get(sam_frame_idx)

            output_filename_base = os.path.splitext(current_frame_filename)[0]

            if frame_specific_masks and isinstance(frame_specific_masks, dict) and len(frame_specific_masks) > 0:
                h, w = img_np.shape[:2]
                combined_mask_for_frame = np.zeros((h, w), dtype=bool)
                valid_masks_found = False
                
                for obj_id, mask_data in frame_specific_masks.items():
                    # mask_data is what's stored from (out_mask_logits[i] > 0).cpu().numpy()
                    # It's likely (1, H, W) or similar. We need (H, W).
                    squeezed_mask = np.squeeze(mask_data)
                    
                    if squeezed_mask.ndim == 2 and squeezed_mask.shape == (h, w):
                        combined_mask_for_frame = np.logical_or(combined_mask_for_frame, squeezed_mask.astype(bool))
                        valid_masks_found = True
                    else:
                        original_mask_shape = mask_data.shape
                        squeezed_mask_shape = squeezed_mask.shape
                        print(f"Warning: Mask for object {obj_id} in frame {sam_frame_idx} (file: {current_frame_filename}) "
                              f"has original shape {original_mask_shape}. After squeeze, shape is {squeezed_mask_shape}. "
                              f"Expected a 2D mask of shape {(h,w)}. Skipping this mask.")
                
                if valid_masks_found:
                    output_save_path = os.path.join(args.output_dir, f"{output_filename_base}_overlay.png")
                    save_mask_overlay(img_np, combined_mask_for_frame, output_save_path)
                else:
                    print(f"No valid masks to overlay for frame {current_frame_filename} (all masks had shape mismatch or no masks). Saving original.")
                    output_save_path = os.path.join(args.output_dir, f"{output_filename_base}_original.png")
                    img_pil.save(output_save_path)
            else:
                print(f"No SAM masks found for frame {current_frame_filename}. Saving original.")
                output_save_path = os.path.join(args.output_dir, f"{output_filename_base}_original.png")
                img_pil.save(output_save_path)

if __name__ == "__main__":
    main()
