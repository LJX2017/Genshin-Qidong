import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
import Utils
from ExampleModel import ExampleModel

TEST_LIGHT_MAP_COUNT = 1


def train_model(model, train_data, args, device):
    total_samples = train_data.shape[0]
    steps_per_epoch = (total_samples + args.batch_size - 1) // args.batch_size
    total_steps = args.epochs * steps_per_epoch

    print(f"   -> Training: {args.epochs} Epochs | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"   -> Loss: MSE+L1 (ratio={args.mse_weight:.2f}) | Warmup: {args.warmup_steps} steps | Gradient Clip: {args.grad_clip}")

    model.train()
    
    # 混合损失函数: MSE + L1
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    # 学习率调度: Warmup + CosineAnnealing
    if args.warmup_steps > 0 and args.warmup_steps < total_steps:
        def lr_lambda(step):
            if step < args.warmup_steps:
                # Warmup阶段: 线性增长
                return step / args.warmup_steps
            else:
                # Cosine退火阶段
                progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # 无warmup或warmup步数过大时，使用纯CosineAnnealing
        if args.warmup_steps >= total_steps:
            print(f"   -> Warning: warmup_steps ({args.warmup_steps}) >= total_steps ({total_steps}), using CosineAnnealing only")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.min_lr
        )

    # Initialize GradScaler for TCNN Mixed Precision
    scaler = torch.amp.GradScaler('cuda')

    global_step = 0
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
                targets_log = torch.log1p(targets)
                
                # 混合损失: MSE + L1
                loss_mse = criterion_mse(pred, targets_log)
                loss_l1 = criterion_l1(pred, targets_log)
                loss = args.mse_weight * loss_mse + (1 - args.mse_weight) * loss_l1

            scaler.scale(loss).backward()
            
            # 梯度裁剪: 防止梯度爆炸
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch + 1:02d}/{args.epochs} | Loss: {epoch_loss / steps_per_epoch:.6f} | LR: {current_lr:.2e}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=262144, help="批次大小")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--lr", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--dataset", type=str, default='/mnt/c/HPRC_Test1/Data/Data_HPRC', help="数据集路径")
    
    # 训练优化参数
    parser.add_argument("--mse_weight", type=float, default=0.5, help="MSE损失权重 (0-1), L1权重=1-mse_weight")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="学习率预热步数 (0表示关闭)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值 (0表示关闭)")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="权重衰减系数")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="最小学习率")
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Using device: {device}")

    os.makedirs(f"./Parameters", exist_ok=True)
    os.makedirs(f"./ResultImages", exist_ok=True)

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

        resolution = lightmap['resolution']
        H, W = resolution['height'], resolution['width']
        lightmap_names = lightmap['lightmaps']
        mask_names = lightmap['masks']

        # 直接从文件加载数据，不使用 cache
        print("Loading data from files...")
        
        # 读取所有时间点的 lightmap 数据
        lightmap_in_different_time = []
        for time_idx, t_str in enumerate(times):
            lightmap_path1 = os.path.join(args.dataset, "Data")
            lightmap_path = os.path.join(lightmap_path1, lightmap_names[t_str])
            if not os.path.exists(lightmap_path):
                raise FileNotFoundError(f"Lightmap file not found: {lightmap_path}")
            lightmap_bin = np.fromfile(lightmap_path, dtype=np.float32)
            lightmap_in_different_time.append(lightmap_bin.reshape(H, W, 3))
        
        # 读取所有时间点的 mask 数据
        mask_in_different_time = []
        for time_idx, t_str in enumerate(times):
            mask_path1 = os.path.join(args.dataset, "Data")
            mask_path = os.path.join(mask_path1, mask_names[t_str])
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            mask_bin = np.fromfile(mask_path, dtype=np.int8)
            mask_in_different_time.append(mask_bin.reshape(H, W))
        
        # 转换为 torch tensor
        lightmap_data = torch.from_numpy(
            np.concatenate([lm.reshape(-1, 3) for lm in lightmap_in_different_time], axis=0)
        ).to(torch.float32).to(device)
        
        mask_data = np.stack(mask_in_different_time, axis=0)  # [T, H, W]
        
        # 构建训练数据：生成归一化坐标并组合时间信息
        coords_list = []
        rgb_list = []
        
        for time_idx, t_str in enumerate(times):
            t_hour = float(t_str) / 100.0
            t_norm = t_hour / 24.0
            
            # 生成归一化坐标
            ys, xs = np.meshgrid(
                np.arange(H, dtype=np.float32),
                np.arange(W, dtype=np.float32),
                indexing='ij'
            )
            x_norm = xs / max(W - 1, 1)
            y_norm = ys / max(H - 1, 1)
            
            # 组合坐标和时间 [H, W, 3]
            coords = np.stack(
                [y_norm, x_norm, np.full_like(x_norm, t_norm)],
                axis=-1
            ).reshape(-1, 3)
            
            # RGB 数据 [H, W, 3]
            rgb = lightmap_in_different_time[time_idx].reshape(-1, 3)
            
            coords_list.append(coords)
            rgb_list.append(rgb)
        
        # 组合所有时间点的数据
        total_coords = np.concatenate(coords_list, axis=0)  # [T*H*W, 3]
        total_rgb = np.concatenate(rgb_list, axis=0)  # [T*H*W, 3]
        
        # 组合成训练数据 [N, 6]: 前3维是坐标，后3维是RGB
        total_data = np.concatenate([total_coords, total_rgb], axis=1)  # [T*H*W, 6]
        total_data = torch.from_numpy(total_data).to(torch.float32).to(device)
        
        # 为评估准备完整的 lightmap_data（包含补帧）
        # 补一帧全0，保持与 time_count 对齐
        lightmap_data_full = torch.cat([
            lightmap_data,
            torch.zeros((H * W, 3), device=device)
        ], dim=0)  # [(T+1)*H*W, 3]
        
        # 为 mask_data 也补一帧
        mask_data_full = np.concatenate([
            mask_data,
            np.zeros((1, H, W), dtype=np.int8)
        ], axis=0)  # [T+1, H, W]

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

            # 按时间逐帧生成坐标并推理（最后一帧为补帧，全 0）
            for t_idx in range(time_count):
                if t_idx < len(times):
                    t_str = times[t_idx]
                    t_hour = float(t_str) / 100.0
                    t_norm = t_hour / 24.0

                    ys, xs = torch.meshgrid(
                        torch.arange(H, device=device, dtype=torch.float32),
                        torch.arange(W, device=device, dtype=torch.float32),
                        indexing='ij'
                    )
                    x_norm = xs / max(W - 1, 1)
                    y_norm = ys / max(H - 1, 1)

                    coords = torch.stack(
                        [y_norm, x_norm, torch.full_like(x_norm, t_norm)],
                        dim=-1
                    ).view(-1, 3).contiguous()

                    for i in range(0, coords.shape[0], eval_bs):
                        batch = coords[i: i + eval_bs]
                        p = torch.clamp(torch.expm1(model(batch)), min=0.0)
                        pred_list.append(p)
                else:
                    # 补一帧全 0，保持与 lightmap_data 对齐
                    pred_list.append(torch.zeros((H * W, 3), device=device))

            pred = torch.cat(pred_list, dim=0)
            pred = pred.reshape(time_count, H, W, 3).permute(0, 3, 1, 2)
            gt_data = lightmap_data_full.reshape(time_count, H, W, 3).permute(0, 3, 1, 2)

            psnr_list = []
            ssim_list = []
            lpips_list = []

            part_size = 256
            rows = (gt_data.shape[2] + part_size - 1) // part_size
            cols = (gt_data.shape[3] + part_size - 1) // part_size

            print("Calculating metrics...")
            for time_idx in range(time_count):
                pred[time_idx, :, mask_data_full[time_idx] <= 0] = 0
                for i in range(rows):
                    for j in range(cols):
                        start_row = i * part_size
                        end_row = min((i + 1) * part_size, gt_data.shape[2])
                        start_col = j * part_size
                        end_col = min((j + 1) * part_size, gt_data.shape[3])

                        gt_part = gt_data[[time_idx], :, start_row:end_row, start_col:end_col]
                        rec_part = pred[[time_idx], :, start_row:end_row, start_col:end_col]
                        mask_part = mask_data_full[time_idx, start_row:end_row, start_col:end_col]
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