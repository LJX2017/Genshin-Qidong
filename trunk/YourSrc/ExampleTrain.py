import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
import Utils
from ExampleModel import ExampleModel

TEST_LIGHT_MAP_COUNT = 10


def train_model(model, train_data, args, device):
    total_samples = train_data.shape[0]
    steps_per_epoch = (total_samples + args.batch_size - 1) // args.batch_size

    print(f"   -> Training: {args.epochs} Epochs | Batch: {args.batch_size} | LR: {args.lr}")

    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch, eta_min=1e-5)

    # Initialize GradScaler for TCNN Mixed Precision
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(args.epochs):
        idx = torch.randperm(total_samples)
        epoch_loss = 0.0

        for i in range(0, total_samples, args.batch_size):
            batch_idx = idx[i: min(i + args.batch_size, total_samples)]
            batch_data = train_data[batch_idx]

            # TCNN requires contiguous inputs
            inputs = batch_data[:, :3].contiguous()
            targets = batch_data[:, 3:].contiguous()

            optimizer.zero_grad()

            # Autocast for TCNN
            with torch.amp.autocast('cuda'):
                pred = model(inputs)
                # Log Space Loss for HDR
                loss = criterion(pred, torch.log1p(targets))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch + 1:02d}/{args.epochs} | Loss: {epoch_loss / steps_per_epoch:.6f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=262144)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--dataset", type=str, default='../Data/Data_HPRC')
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Using device: {device}")

    os.makedirs(f"./Parameters", exist_ok=True)
    os.makedirs(f"./ResultImages", exist_ok=True)
    os.makedirs(f"./Cache", exist_ok=True)

    with open(os.path.join(args.dataset, 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    target_config_lightmaps = config['lightmap_list'][:TEST_LIGHT_MAP_COUNT]

    times = ["0", "100", "200", "300", "400", "500", "590", "600", "700", "800", "900", "1000", "1100", "1200", "1300",
             "1400", "1500", "1600", "1700", "1800", "1810", "1900", "2000", "2100", "2200", "2300"]
    time_count = len(times) + 1

    total_psnr = []
    total_ssim = []
    total_lpips = []

    for lightmap in target_config_lightmaps:
        id = lightmap['id']
        print(f"\n>>> Processing {lightmap['level']}_{id} <<<")

        cache_path = f"./Cache/processed_{lightmap['level']}_{id}.pt"
        resolution = lightmap['resolution']

        if os.path.exists(cache_path):
            # weights_only=False is required for complex dicts
            cached = torch.load(cache_path, weights_only=False)
            total_data = cached['total_data'].to(device)
            mask_data = cached['mask_data']
            lightmap_data = cached['lightmap_data'].to(device)
            total_coords = cached['total_coords'].to(device)
        else:
            print("Cache missing! Please copy loading logic if needed.")
            return

        # Train
        model = ExampleModel(hidden_dim=args.hidden_dim).to(device)
        model = train_model(model, total_data, args, device)

        # Save Parameters (Float32 Baseline)
        print("Saving raw binary parameters...")
        all_params = []
        for param in model.parameters():
            all_params.append(param.detach().cpu().numpy().flatten())

        params_array = np.concatenate(all_params).astype(np.float32)

        save_path = f"./Parameters/model_{lightmap['level']}_{id}_params.bin"
        params_array.tofile(save_path)
        print(f"Saved {len(params_array) * 4 / 1024 / 1024:.2f} MB")

        # Evaluation
        with torch.no_grad():
            model.eval()
            pred_list = []
            eval_bs = 262144

            for i in range(0, total_coords.shape[0], eval_bs):
                batch = total_coords[i: i + eval_bs].contiguous()
                p = torch.clamp(torch.expm1(model(batch[:, :3])), min=0.0)
                pred_list.append(p)

            pred = torch.cat(pred_list, dim=0)
            pred = pred.reshape(time_count, resolution['height'], resolution['width'], 3).permute(0, 3, 1, 2)
            gt_data = lightmap_data.reshape(time_count, resolution['height'], resolution['width'], 3).permute(0, 3, 1,
                                                                                                              2)

            psnr_list = []
            ssim_list = []
            lpips_list = []

            part_size = 256
            rows = (gt_data.shape[2] + part_size - 1) // part_size
            cols = (gt_data.shape[3] + part_size - 1) // part_size

            print("Calculating metrics...")
            for time_idx in range(time_count):
                pred[time_idx, :, mask_data[time_idx] <= 0] = 0
                for i in range(rows):
                    for j in range(cols):
                        start_row = i * part_size
                        end_row = min((i + 1) * part_size, gt_data.shape[2])
                        start_col = j * part_size
                        end_col = min((j + 1) * part_size, gt_data.shape[3])

                        gt_part = gt_data[[time_idx], :, start_row:end_row, start_col:end_col]
                        rec_part = pred[[time_idx], :, start_row:end_row, start_col:end_col]
                        mask_part = mask_data[time_idx, start_row:end_row, start_col:end_col]
                        valid_mask = mask_part >= 127

                        if np.any(valid_mask) and gt_part.max() != 0:
                            psnr_list.append(Utils.cal_psnr(gt_part, rec_part, mask_part))
                            ssim_list.append(Utils.cal_ssim(gt_part, rec_part))
                            lpips_list.append(Utils.cal_lpips(gt_part, rec_part))

            # REPORT ALL 3 METRICS
            local_psnr = np.mean(psnr_list)
            local_ssim = np.mean(ssim_list)
            local_lpips = np.mean(lpips_list)

            print(f"Metrics for {lightmap['level']}_{id}:")
            print(f"  PSNR:  {local_psnr:.2f}")
            print(f"  SSIM:  {local_ssim:.4f}")
            print(f"  LPIPS: {local_lpips:.4f}")

            total_psnr.extend(psnr_list)
            total_ssim.extend(ssim_list)
            total_lpips.extend(lpips_list)

    print(f"\n========================================")
    print(f"FINAL DATASET METRICS")
    print(f"Mean PSNR:  {np.mean(total_psnr):.2f}")
    print(f"Mean SSIM:  {np.mean(total_ssim):.4f}")
    print(f"Mean LPIPS: {np.mean(total_lpips):.4f}")
    print(f"========================================")


if __name__ == "__main__":
    main()