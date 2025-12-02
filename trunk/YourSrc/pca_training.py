import argparse
import os
import json
import numpy as np
import torch

import Utils
from pca_model import PCALightmapModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../Data/Data_HPRC")
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--K", type=int, default=150)
    parser.add_argument("--max_lightmaps", type=int, default=5,
                        help="Limit number of lightmaps to train on (-1 = all)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("./Parameters", exist_ok=True)

    config_path = os.path.join(args.dataset, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lightmap_list = cfg["lightmap_list"]
    num_total = len(lightmap_list)
    if args.max_lightmaps > 0:
        num_use = min(args.max_lightmaps, num_total)
    else:
        num_use = num_total

    print(f"Total lightmaps in config: {num_total}")
    print(f"Training PCA on {num_use} lightmaps")
    print(f"Official setting: block_size={args.block_size}, K={args.K}, coeffs=int8")

    # Global metrics across all lightmaps
    total_psnr = []
    total_ssim = []
    total_lpips = []

    for lm_idx, lm_cfg in enumerate(lightmap_list[:num_use]):
        level = lm_cfg["level"]
        lm_id = lm_cfg["id"]
        res = lm_cfg["resolution"]
        H, W = res["height"], res["width"]

        time_keys = sorted(lm_cfg["lightmaps"].keys(), key=lambda s: int(s))
        times_int = [int(k) for k in time_keys]
        T = len(times_int)

        print(f"\n>>> Processing lightmap {level}_{lm_id} (index {lm_idx})")
        print(f"Resolution: {H}x{W}, T={T}")

        # ----------------------------------------------------------
        # 1) Load frames and masks
        # ----------------------------------------------------------
        frames_rgb = {}
        frames_mask = {}
        for t_str in time_keys:
            t = int(t_str)
            lm_path = os.path.join(args.dataset, "Data", lm_cfg["lightmaps"][t_str])
            mk_path = os.path.join(args.dataset, "Data", lm_cfg["masks"][t_str])

            frames_rgb[t] = np.fromfile(lm_path, dtype=np.float32).reshape(H, W, 3)
            frames_mask[t] = np.fromfile(mk_path, dtype=np.int8).reshape(H, W)

        # ----------------------------------------------------------
        # 2) Build 4x4 blocks over all valid grid positions
        #    (no skipping: we assume resolution divisible by block_size)
        # ----------------------------------------------------------
        B = args.block_size
        assert H % B == 0 and W % B == 0, "Resolution must be divisible by block_size"

        grid_h = H // B
        grid_w = W // B
        N_blocks = grid_h * grid_w
        D = B * B * 3 * T

        print(f"Block grid: {grid_h} x {grid_w} => N_blocks={N_blocks}, D={D}")

        blocks = np.zeros((N_blocks, D), dtype=np.float32)

        # Flattening order: [by, bx, y_in_block, x_in_block, channel, time]
        # Then reshape to vector of length D.
        idx = 0
        for gy in range(grid_h):
            by = gy * B
            for gx in range(grid_w):
                bx = gx * B

                blk = np.zeros((B, B, 3, T), dtype=np.float32)
                for ti, t in enumerate(times_int):
                    blk[..., ti] = frames_rgb[t][by:by + B, bx:bx + B, :]

                blocks[idx] = blk.reshape(-1)
                idx += 1

        # ----------------------------------------------------------
        # 3) Train PCA model
        # ----------------------------------------------------------
        model = PCALightmapModel(block_size=B, K=args.K)
        model.fit(blocks, H=H, W=W, T=T)

        # ----------------------------------------------------------
        # 4) Save parameters (binary)
        # ----------------------------------------------------------
        param_path = os.path.join(
            "./Parameters",
            f"pca_model_{level}_{lm_id}_params.bin"
        )
        model.save(param_path)
        print(f"Saved PCA params to: {param_path}")

        # ----------------------------------------------------------
        # 5) Reconstruct and evaluate metrics
        # ----------------------------------------------------------
        print("Reconstructing for evaluation...")

        blocks_dec = model.decode_blocks()  # [N_blocks, B, B, 3, T]
        recon_frames = {
            t: np.zeros_like(frames_rgb[t], dtype=np.float32)
            for t in times_int
        }

        for idx in range(N_blocks):
            gy = idx // grid_w
            gx = idx % grid_w
            by = gy * B
            bx = gx * B

            blk = blocks_dec[idx]  # (B, B, 3, T)
            for ti, t in enumerate(times_int):
                recon_frames[t][by:by + B, bx:bx + B, :] = blk[..., ti]

        # Metrics for this lightmap
        psnr_list = []
        ssim_list = []
        lpips_list = []

        for t in times_int:
            gt = torch.from_numpy(
                frames_rgb[t].transpose(2, 0, 1)
            ).unsqueeze(0).to(torch.float32).to(device)

            rc = torch.from_numpy(
                recon_frames[t].transpose(2, 0, 1)
            ).unsqueeze(0).to(torch.float32).to(device)

            mk = torch.from_numpy(frames_mask[t]).to(device)

            # Use existing Utils metrics (masked PSNR, SSIM, LPIPS)
            psnr_list.append(Utils.cal_psnr(gt, rc, mk))
            ssim_list.append(Utils.cal_ssim(gt, rc))
            lpips_list.append(Utils.cal_lpips(gt, rc))

        lm_psnr = float(np.mean(psnr_list))
        lm_ssim = float(np.mean(ssim_list))
        lm_lpips = float(np.mean(lpips_list))

        print(f"Metrics for lightmap {level}_{lm_id}:")
        print(f"  PSNR:  {lm_psnr:.4f}")
        print(f"  SSIM:  {lm_ssim:.4f}")
        print(f"  LPIPS: {lm_lpips:.4f}")

        total_psnr.extend(psnr_list)
        total_ssim.extend(ssim_list)
        total_lpips.extend(lpips_list)

    # --------------------------------------------------------------
    # Global dataset metrics
    # --------------------------------------------------------------
    print("\n================ GLOBAL METRICS ================")
    if total_psnr:
        print(f"PSNR (all lightmaps):  {float(np.mean(total_psnr)):.4f}")
        print(f"SSIM (all lightmaps):  {float(np.mean(total_ssim)):.4f}")
        print(f"LPIPS (all lightmaps): {float(np.mean(total_lpips)):.4f}")
    else:
        print("No lightmaps processed.")
    print("================================================")


if __name__ == "__main__":
    main()
