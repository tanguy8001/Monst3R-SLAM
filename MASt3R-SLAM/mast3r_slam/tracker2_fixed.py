import torch
import os
import PIL.Image
import numpy as np

from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import (
    mast3r_match_asymmetric,
)
from mast3r_slam.dynamic_mask_utils import (
    get_dynamic_mask,
    build_sam2_video_predictor,
    SAM2_CHECKPOINT_DEFAULT,
    SAM2_MODEL_CONFIG_NAME_FOR_HYDRA,
    SAM2_MODEL_CONFIG_ABSOLUTE_PATH,
)
from mast3r_slam.inpainting_utils import InpaintingPipeline
from thirdparty.monst3r.third_party.raft import load_RAFT


class FrameTracker2:
    def __init__(self, mast3r, frames, device):
        self.cfg = config["tracking"]
        self.mast3r = mast3r

        current_dir = os.path.dirname(os.path.abspath(__file__))
        raft_weights_path = os.path.join(current_dir, "../thirdparty/monst3r/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
        raft_weights_path = os.path.normpath(raft_weights_path)
        self.raft_model = load_RAFT(raft_weights_path)
        self.raft_model = self.raft_model.to(device)
        self.raft_model.eval()
        
        self.keyframes = frames
        self.device = device

        self.sam2_predictor = None
        if config.get("use_dynamic_mask", False) and config.get("refine_dynamic_mask_with_sam2", True):
            try:
                print("Initializing SAM2 predictor for tracker...")
                if not os.path.exists(SAM2_CHECKPOINT_DEFAULT):
                     print(f"Warning: SAM2 checkpoint not found at {SAM2_CHECKPOINT_DEFAULT}. SAM2 refinement disabled.")
                elif not os.path.exists(SAM2_MODEL_CONFIG_ABSOLUTE_PATH):
                     print(f"Warning: SAM2 config not found at {SAM2_MODEL_CONFIG_ABSOLUTE_PATH}. SAM2 refinement disabled.")
                else:
                    self.sam2_predictor = build_sam2_video_predictor(
                        SAM2_MODEL_CONFIG_NAME_FOR_HYDRA,
                        SAM2_CHECKPOINT_DEFAULT,
                        device=device
                    )
                    self.sam2_predictor.eval()
                    print("SAM2 predictor initialized.")
            except Exception as e:
                print(f"Error initializing SAM2 predictor: {e}. SAM2 refinement disabled.")
                self.sam2_predictor = None

        # Initialize inpainting pipeline if enabled
        self.inpainting_pipeline = None
        if config.get("use_inpainting", True):  # Enable by default
            try:
                print("Initializing inpainting pipeline...")
                self.inpainting_pipeline = InpaintingPipeline(
                    sam_model_type=config.get("inpainting_sam_model", "vit_b"),
                    sam_ckpt=config.get("inpainting_sam_ckpt", "thirdparty/inpaint/pretrained_models/sam_vit_b_01ec64.pth"),
                    lama_config=config.get("inpainting_lama_config", "thirdparty/inpaint/lama/configs/prediction/default.yaml"),
                    lama_ckpt=config.get("inpainting_lama_ckpt", "thirdparty/inpaint/pretrained_models/big-lama"),
                    device=device
                )
                print("Inpainting pipeline initialized successfully.")
            except Exception as e:
                print(f"Error initializing inpainting pipeline: {e}. Inpainting disabled.")
                self.inpainting_pipeline = None

        self.reset_idx_f2k()

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None

    def track(self, frame: Frame):
        keyframe = self.keyframes.last_keyframe()

        h, w = frame.img.shape[-2:]

        if config.get("use_dynamic_mask", False):
            try:
                computed_mask, point_prompts = get_dynamic_mask(
                    self.mast3r, self.raft_model, frame, keyframe,
                    threshold=config.get("dynamic_mask_threshold", 0.35),
                    refine_with_sam2=config.get("refine_dynamic_mask_with_sam2", True),
                    sam2_predictor=self.sam2_predictor,
                    only_dynamic_points=config.get("only_dynamic_points", True)
                )
                frame.dynamic_mask = computed_mask
                frame.point_prompts = point_prompts
                
                if computed_mask.shape[0] == h and computed_mask.shape[1] == w:
                    if computed_mask.any():
                        print(f"Dynamic mask computed with dynamic regions for frame {frame.frame_id}.")
                    
                    # Option 1: Use inpainting if available and point prompts exist
                    if (config.get("use_inpainting", True) and 
                        self.inpainting_pipeline is not None and 
                        point_prompts and len(point_prompts) > 0):
                        
                        print(f"Inpainting dynamic regions for frame {frame.frame_id} using {len(point_prompts)} point prompts.")
                        
                        # Convert frame.img to numpy format expected by inpainting
                        # frame.img is typically (1, 3, H, W) in [-1, 1] range
                        frame_img_np = frame.uimg.cpu().numpy()  # Use uimg which is (H, W, 3) in [0, 1]
                        frame_img_np = (frame_img_np * 255).astype(np.uint8)  # Convert to [0, 255] uint8
                        
                        # Perform inpainting
                        inpainted_img_np, inpaint_mask = self.inpainting_pipeline.inpaint_frame_with_points(
                            frame_img_np, 
                            point_prompts,
                            dilate_kernel_size=config.get("inpainting_dilate_kernel", 15)
                        )
                        
                        # Convert back to tensor format and update frame.img
                        inpainted_img_normalized = inpainted_img_np.astype(np.float32) / 255.0  # Back to [0, 1]
                        inpainted_img_tensor = torch.from_numpy(inpainted_img_normalized).to(frame.img.device)
                        inpainted_img_tensor = inpainted_img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                        inpainted_img_tensor = (inpainted_img_tensor * 2.0) - 1.0  # Convert to [-1, 1] range
                        
                        frame.img = inpainted_img_tensor
                        frame.inpaint_mask = torch.from_numpy(inpaint_mask).to(frame.img.device)
                        
                        print(f"Successfully inpainted frame {frame.frame_id}.")
                        
                    # Option 2: Fallback to mask-based inpainting if point prompts don't work
                    elif (config.get("use_inpainting", True) and 
                          self.inpainting_pipeline is not None and 
                          computed_mask.any()):
                        
                        print(f"Inpainting dynamic regions for frame {frame.frame_id} using mask fallback.")
                        
                        # Convert frame.img to numpy format
                        frame_img_np = frame.uimg.cpu().numpy()
                        frame_img_np = (frame_img_np * 255).astype(np.uint8)
                        
                        # Convert mask to numpy
                        mask_np = computed_mask.cpu().numpy()
                        
                        # Perform mask-based inpainting
                        inpainted_img_np = self.inpainting_pipeline.inpaint_frame_with_mask(
                            frame_img_np,
                            mask_np,
                            dilate_kernel_size=config.get("inpainting_dilate_kernel", 15)
                        )
                        
                        # Convert back to tensor format
                        inpainted_img_normalized = inpainted_img_np.astype(np.float32) / 255.0
                        inpainted_img_tensor = torch.from_numpy(inpainted_img_normalized).to(frame.img.device)
                        inpainted_img_tensor = inpainted_img_tensor.permute(2, 0, 1).unsqueeze(0)
                        inpainted_img_tensor = (inpainted_img_tensor * 2.0) - 1.0
                        
                        frame.img = inpainted_img_tensor
                        
                        print(f"Successfully inpainted frame {frame.frame_id} using mask fallback.")
                    
                    # Option 3: Fallback to black masking (original behavior)
                    else:
                        print(f"Using black masking fallback for frame {frame.frame_id}.")
                        masked_input_img_tensor = frame.img.clone()
                        expanded_mask = computed_mask.unsqueeze(0).expand_as(masked_input_img_tensor)
                        masked_input_img_tensor[expanded_mask] = 0.0  # Mask with black
                        frame.img = masked_input_img_tensor
                        
                else:
                    print(f"Warning: Computed dynamic mask shape {computed_mask.shape} mismatch with frame image shape {(h,w)}. No processing applied.")
                    frame.dynamic_mask = torch.zeros((h, w), dtype=torch.bool, device=frame.img.device)
                    frame.point_prompts = []
                    
            except Exception as e:
                print(f"Failed to process dynamic content for frame {frame.frame_id}: {str(e)}")
                print("Continuing without dynamic processing. Using fallback all-static mask.")
                frame.dynamic_mask = torch.zeros((h, w), dtype=torch.bool, device=frame.img.device)
                frame.point_prompts = []
        else:
            frame.dynamic_mask = torch.zeros((h, w), dtype=torch.bool, device=frame.img.device)
            frame.point_prompts = []


        # --- Debug: Save inpainting results ---
        if (config.get("debug_save_inpainting", False) and 
            hasattr(frame, 'inpaint_mask') and frame.inpaint_mask is not None):
            try:
                original_img = frame.uimg.cpu().numpy()
                inpainted_img = (frame.img.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
                
                if original_img.shape[0] == 3:
                    original_img = np.transpose(original_img, (1, 2, 0))
                
                # Create comparison image
                comparison = np.concatenate([original_img, inpainted_img], axis=1)
                comparison_pil = PIL.Image.fromarray((comparison * 255).astype(np.uint8))
                
                dataset_name = config.get("dataset", {}).get("name", "unknown_dataset")
                video_name = config.get("dataset", {}).get("sequence", config.get("dataset", {}).get("video", "unknown_video"))
                save_dir = os.path.join("logs", dataset_name, video_name, "debug_inpainting")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"frame_{frame.frame_id:06d}_inpainted.png")
                comparison_pil.save(save_path)
            except Exception as e:
                print(f"Error saving inpainting debug for frame {frame.frame_id}: {e}")
        #--- End Debug ---

        # === Matching process ===
        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            model=self.mast3r, frame_i=frame, frame_j=keyframe, idx_i2j_init=self.idx_f2k
        )
        print(f"Shape of valid_match_k: {valid_match_k.shape}")
        print(f"Shape of frame.img: {frame.img.shape}")

        self.idx_f2k = idx_f2k.clone()
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]
        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # --- Debug: Save dynamic mask overlay ---
        if config.get("debug_save_dynamic_mask", True) and frame.dynamic_mask is not None:
            try:
                img_to_save = frame.uimg.cpu().numpy()
                mask_to_save = frame.dynamic_mask.cpu().numpy()
                if img_to_save.shape[0] == 3:
                    img_to_save = np.transpose(img_to_save, (1, 2, 0))
                img_pil = PIL.Image.fromarray((img_to_save * 255).astype(np.uint8))
                img_pil = img_pil.convert("RGBA")
                overlay_color = (255, 0, 0, 128)
                overlay = PIL.Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
                mask_pixels = np.zeros((*mask_to_save.shape, 4), dtype=np.uint8)
                mask_pixels[mask_to_save] = overlay_color
                mask_image = PIL.Image.fromarray(mask_pixels, 'RGBA')
                overlay.paste(mask_image, (0, 0), mask_image)
                img_with_mask = PIL.Image.alpha_composite(img_pil, overlay)
                img_with_mask = img_with_mask.convert("RGB")
                dataset_name = config.get("dataset", {}).get("name", "unknown_dataset")
                video_name = config.get("dataset", {}).get("sequence", config.get("dataset", {}).get("video", "unknown_video"))
                save_dir = os.path.join("logs", dataset_name, video_name, "debug_dynamic_mask")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"frame_{frame.frame_id:06d}.png")
                img_with_mask.save(save_path)
            except Exception as e:
                print(f"Error saving dynamic mask overlay for frame {frame.frame_id}: {e}")
        #--- End Debug ---

        frame.update_pointmap(Xff, Cff) # Eq. (9) in paper. Populates frame.X_canon, frame.C_canon

        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:] # Use potentially restored frame.img shape
        if use_calib:
            K = keyframe.K
        else:
            K = None

        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt_base = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_opt = valid_opt_base.clone()
        
        # --- Debug: Save final valid_opt mask ---
        if config.get("debug_save_final_valid_opt_mask", True):
            try:
                mask_for_saving = valid_opt.squeeze().reshape(h, w) 
                img_to_save = frame.uimg.cpu().numpy() 
                mask_to_save_np = mask_for_saving.cpu().numpy()
                if img_to_save.shape[0] == 3: 
                    img_to_save = np.transpose(img_to_save, (1, 2, 0))
                img_pil = PIL.Image.fromarray((img_to_save * 255).astype(np.uint8))
                img_pil = img_pil.convert("RGBA")
                overlay_color = (0, 255, 0, 128) 
                overlay = PIL.Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
                mask_pixels = np.zeros((*mask_to_save_np.shape, 4), dtype=np.uint8)
                mask_pixels[mask_to_save_np] = overlay_color
                mask_image = PIL.Image.fromarray(mask_pixels, 'RGBA')
                overlay.paste(mask_image, (0, 0), mask_image)
                img_with_mask = PIL.Image.alpha_composite(img_pil, overlay)
                img_with_mask = img_with_mask.convert("RGB")
                dataset_name = config.get("dataset", {}).get("name", "unknown_dataset")
                video_name = config.get("dataset", {}).get("sequence", config.get("dataset", {}).get("video", "unknown_video"))
                save_dir = os.path.join("logs", dataset_name, video_name, "debug_final_valid_opt_mask")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"frame_{frame.frame_id:06d}_final_opt.png")
                img_with_mask.save(save_path)
            except Exception as e:
                print(f"Error saving final valid_opt mask overlay for frame {frame.frame_id}: {e}")
        # --- End Debug ---

        valid_kf = valid_match_k & valid_Q

        match_frac = valid_opt.sum() / valid_opt.numel()
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True

        try:
            # Track
            if not use_calib:
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                )
        except Exception as e:
            print(f"Cholesky failed {frame.frame_id}")
            return False, [], True

        frame.T_WC = T_WCf

        # Use pose to transform points to update keyframe
        Xkk = T_CkCf.act(Xkf)
        keyframe.update_pointmap(Xkk, Ckf)
        # write back the fitered pointmap
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # Keyframe selection
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()
        unique_frac_f = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]

        # Rest idx if new keyframe
        if new_kf:        
            self.reset_idx_f2k()

        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )

    def get_points_poses(self, frame, keyframe, idx_f2k, img_size, use_calib, K=None):
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        # Average confidence
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib:
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # Setup pixel coordinates
            uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # Avoid any bad calcs in log
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J):
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        # Precalculate distance and ray for obs k
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # r = z-h(x)
            r = rd_k - rd_f_Ck
            # Jacobian
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf

    def opt_pose_calib_sim3(
        self, Xf, Xk, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size
    ):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = valid2 * sqrt_info

            # r = z-h(x)
            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf 