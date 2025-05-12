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
from mast3r_slam.monst3r_utils import monst3r_match_asymmetric, get_dynamic_mask
from thirdparty.monst3r.third_party.raft import load_RAFT


class FrameTracker2:
    def __init__(self, mast3r, monst3r, frames, device):
        self.cfg = config["tracking"]
        self.mast3r = mast3r
        self.monst3r = monst3r

        current_dir = os.path.dirname(os.path.abspath(__file__))
        raft_weights_path = os.path.join(current_dir, "../thirdparty/monst3r/third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
        raft_weights_path = os.path.normpath(raft_weights_path)
        self.raft_model = load_RAFT(raft_weights_path)
        self.raft_model = self.raft_model.to(device)
        self.raft_model.eval()
        
        self.keyframes = frames
        self.device = device

        self.reset_idx_f2k()

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None

    def track(self, frame: Frame):
        keyframe = self.keyframes.last_keyframe()

        # valid_match_k is a boolean mask of the points that are matched between the frame and the keyframe
        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = monst3r_match_asymmetric(
            mast3r=self.mast3r, monst3r=self.monst3r, frame_i=frame, frame_j=keyframe, idx_i2j_init=self.idx_f2k
        )
        # Save idx for next
        self.idx_f2k = idx_f2k.clone()

        # Get rid of batch dim
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # Detect dynamic objects if enabled in config
        dynamic_mask_computed = False
        is_dynamic_point = None
        if config.get("use_dynamic_mask", False):
            try:
                dynamic_mask = get_dynamic_mask(
                    self.monst3r, self.raft_model,frame, keyframe,
                    threshold=config.get("dynamic_mask_threshold", 0.35),
                    refine_with_sam2=config.get("refine_dynamic_mask_with_sam2", True)
                )
                # Use dynamic mask to filter out points on moving objects
                # Reshape dynamic_mask to match our points
                h, w = frame.img.shape[-2:] # Use frame's shape directly
                # Check if the returned mask is valid (not all zeros)
                if dynamic_mask.any():
                    reshaped_mask = dynamic_mask.reshape(-1)
                    # Identify points corresponding to dynamic pixels
                    is_dynamic_point = reshaped_mask[idx_f2k] > 0.5 # Boolean mask for points
                    dynamic_mask_computed = True

                    # Store dynamic mask in frame for visualization
                    frame.dynamic_mask = dynamic_mask
                    print(f"Successfully computed dynamic mask for frame {frame.frame_id}")
                else:
                    # Handle case where dynamic mask computation failed (e.g., no K) or produced an empty mask
                    print(f"Dynamic mask computation resulted in an empty mask for frame {frame.frame_id}.")
                    # Ensure frame has an empty mask placeholder if needed elsewhere
                    frame.dynamic_mask = torch.zeros((h, w), dtype=torch.bool, device=frame.img.device)

            except Exception as e:
                print(f"Failed to compute dynamic mask for frame {frame.frame_id}: {str(e)}")
                print("Continuing without dynamic mask")
                # Create an empty mask as fallback
                h, w = frame.img.shape[-2:]
                frame.dynamic_mask = torch.zeros((h, w), dtype=torch.bool, device=frame.img.device)

        # --- Debug: Save dynamic mask overlay ---
        if config.get("debug_save_dynamic_mask", True): # Set to False to disable saving
             try:
                 # Get necessary data (ensure they are on CPU)
                 img_to_save = frame.uimg.cpu().numpy() # HWC float [0,1]
                 mask_to_save = frame.dynamic_mask.cpu().numpy() # HxW boolean

                 # Convert image to uint8 PIL
                 if img_to_save.shape[0] == 3: # Handle potential CHW case, though uimg should be HWC
                     img_to_save = np.transpose(img_to_save, (1, 2, 0))
                 img_pil = PIL.Image.fromarray((img_to_save * 255).astype(np.uint8))
                 img_pil = img_pil.convert("RGBA") # Ensure RGBA for overlay

                 # Create colored mask overlay (semi-transparent red)
                 # Create an RGBA image for the overlay, starting fully transparent
                 overlay_color = (255, 0, 0, 128) # Red with 50% alpha
                 overlay = PIL.Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
                 # Create a mask image where dynamic pixels are opaque red, others transparent
                 mask_pixels = np.zeros((*mask_to_save.shape, 4), dtype=np.uint8)
                 mask_pixels[mask_to_save] = overlay_color
                 mask_image = PIL.Image.fromarray(mask_pixels, 'RGBA')

                 # Paste the colored mask onto the transparent overlay
                 overlay.paste(mask_image, (0, 0), mask_image) # Use mask_image as its own mask

                 # Combine image and overlay
                 img_with_mask = PIL.Image.alpha_composite(img_pil, overlay)
                 img_with_mask = img_with_mask.convert("RGB") # Convert back to RGB for saving

                 # Define save path
                 # Attempt to get dataset/sequence name from config, provide defaults
                 dataset_name = config.get("dataset", {}).get("name", "unknown_dataset")
                 # Use 'sequence' or 'video' as potential keys for the video name
                 video_name = config.get("dataset", {}).get("sequence", config.get("dataset", {}).get("video", "unknown_video"))
                 save_dir = os.path.join("logs", dataset_name, video_name, "debug_dynamic_mask")
                 os.makedirs(save_dir, exist_ok=True)
                 save_path = os.path.join(save_dir, f"frame_{frame.frame_id:06d}.png")

                 # Save the image
                 img_with_mask.save(save_path)
                 # Optional: print confirmation
                 # print(f"Saved dynamic mask overlay to {save_path}")

             except Exception as e:
                 print(f"Error saving dynamic mask overlay for frame {frame.frame_id}: {e}")
        # --- End Debug ---

        # Update keyframe pointmap after registration (need pose)
        frame.update_pointmap(Xff, Cff)

        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K
        else:
            K = None

        # Get poses and point correspondences and confidences
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # Get valid
        # Use canonical confidence average
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        # Calculate base validity based on confidence and matching
        valid_opt_base = valid_match_k & valid_Cf & valid_Ck & valid_Q

        # Initialize final valid mask
        valid_opt = valid_opt_base.clone()

        # Filter dynamic points
        if dynamic_mask_computed and is_dynamic_point is not None:
            # Ensure shapes match to avoid broadcasting issues
            is_dynamic_point_reshaped = is_dynamic_point.view_as(valid_opt)

            original_valid_count = valid_opt.sum() # Count from valid_opt_base initially
            # Directly filter out all points marked as dynamic by the dynamic_mask module
            valid_opt = valid_opt_base & (~is_dynamic_point_reshaped)
            filtered_count = original_valid_count - valid_opt.sum()

            if filtered_count > 0:
                print(f"Filtered {filtered_count} dynamic points (identified by dynamic_mask module) from optimization for frame {frame.frame_id}")

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
