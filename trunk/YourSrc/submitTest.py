import json
import os
import time
import numpy as np
import torch
import yaml

import PCA_VQ_Interface as Interface
import Utils

PARAMETERS = './pca_vq_parameters'

if __name__ == '__main__':
    # ================= CONFIGURATION =================
    # Set this to an integer to test only the top K lightmaps (e.g., 5).
    # Set to None or 0 to test ALL lightmaps.
    TEST_K = 10
    # =================================================

    # 所有大赛使用的数据集
    data_set_list = ['Data_HPRC']
    test_time = list(range(24)) + [5.9, 18.1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        psnr_best_value = Utils.extract_numeric_value(config['metrics']['psnr']['best_value'])
        psnr_worst_value = Utils.extract_numeric_value(config['metrics']['psnr']['worst_value'])
        ssim_best_value = Utils.extract_numeric_value(config['metrics']['ssim']['best_value'])
        ssim_worst_value = Utils.extract_numeric_value(config['metrics']['ssim']['worst_value'])
        lpips_best_value = Utils.extract_numeric_value(config['metrics']['lpips']['best_value'])
        lpips_worst_value = Utils.extract_numeric_value(config['metrics']['lpips']['worst_value'])
        time_best_value = Utils.extract_numeric_value(config['metrics']['inference time']['best_value'])
        time_worst_value = Utils.extract_numeric_value(config['metrics']['inference time']['worst_value'])
        compression_best_value = Utils.extract_numeric_value(config['metrics']['compression ratio']['best_value'])
        compression_worst_value = Utils.extract_numeric_value(config['metrics']['compression ratio']['worst_value'])

    for data_set in data_set_list:
        # 读取数据集下的配置文件
        dataset_path = f'../../../Data/{data_set}'
        config_file = 'config.json'
        with open(os.path.join(dataset_path, config_file), 'r', encoding='utf-8') as f:
            data = json.load(f)

        # --- MODIFICATION START: Slice the list to Top K ---
        full_list = data['lightmap_list']
        if TEST_K is not None and TEST_K > 0:
            print(f"!!! DEBUG MODE: Limiting test to top {TEST_K} lightmaps !!!")
            data['lightmap_list'] = full_list[:TEST_K]
        # --- MODIFICATION END ---

        lightmap_count = data['lightmap_count']
        total_maps = len(data['lightmap_list'])  # For progress tracking

        # 初始化整个数据集的指标list
        psnr_list = []
        ssim_list = []
        lpips_list = []
        time_list = []
        data_size = 0

        print(f"Starting evaluation on {total_maps} lightmaps...", flush=True)

        # 计算所有lightmap的指标
        for idx, lightmap_config in enumerate(data['lightmap_list']):
            # 从配置文件中获取lightmap的id、lightmap路径、mask路径、分辨率
            lm_id = lightmap_config.get('id', 'unknown')
            level = lightmap_config.get('level', 'unknown')
            lightmap_names = lightmap_config['lightmaps']
            mask_names = lightmap_config['masks']
            resolution = lightmap_config['resolution']

            # --- NEW: Local lists for this specific lightmap ---
            curr_psnr_list = []
            curr_ssim_list = []
            curr_lpips_list = []
            curr_time_list = []

            # 计算数据大小，3通道，24个时刻，每个通道4个字节
            data_size += resolution['height'] * resolution['width'] * 3 * 4 * 24

            # 获取接口对象
            your_interface = Interface.get(lightmap_config, device)

            # 分别重建每个时刻的lightmap
            for current_time in test_time:
                with torch.no_grad():
                    # 计算推理时间
                    torch.cuda.synchronize()
                    time_start = time.time()
                    your_interface.reconstruct(current_time)

                    # 获取重建结果
                    lightmap_reconstruct = your_interface.get_result()

                    torch.cuda.synchronize()
                    time_end = time.time()
                    step_time = time_end - time_start
                    curr_time_list.append(step_time)

                    # 测试Random Access
                    random_coord = np.zeros((1, 3))
                    random_y = np.random.randint(0, resolution['height'])
                    random_x = np.random.randint(0, resolution['width'])
                    random_coord[0, 0] = random_y
                    random_coord[0, 1] = random_x
                    random_coord[0, 2] = current_time
                    random_lightmap_reconstruct = your_interface.random_test(random_coord)
                    if not torch.allclose(random_lightmap_reconstruct[0, :],
                                          lightmap_reconstruct[0, :, random_y, random_x], atol=1e-2):
                        print(f"Random Access Test Failed at {current_time} {random_y} {random_x} !!!", flush=True)
                        exit()

                    # 读取原始lightmap数据计算指标
                    lightmap_path = os.path.join(dataset_path, "Data", lightmap_names[str(int(current_time * 100))])
                    mask_path = os.path.join(dataset_path, "Data", mask_names[str(int(current_time * 100))])

                    lightmap_data = np.fromfile(lightmap_path, dtype=np.float32)
                    mask_data = np.fromfile(mask_path, dtype=np.int8)
                    lightmap = lightmap_data.reshape(resolution['height'], resolution['width'], 3)
                    mask = mask_data.reshape(resolution['height'], resolution['width'])

                    # 每256*256分辨率计算一次指标，最后取平均值
                    # 我们在计算指标的时候不会考虑无效的像素，所以你可以在你的训练中不考虑这些无效像素
                    lightmap_reconstruct[:, :, mask <= 0] = 0
                    lightmap = torch.from_numpy(lightmap).permute(2, 0, 1).unsqueeze(0).to(device)

                    part_size = 256
                    rows = (lightmap.shape[2] + part_size - 1) // part_size
                    cols = (lightmap.shape[3] + part_size - 1) // part_size
                    for i in range(rows):
                        for j in range(cols):
                            start_row = i * part_size
                            end_row = min((i + 1) * part_size, lightmap.shape[2])
                            start_col = j * part_size
                            end_col = min((j + 1) * part_size, lightmap.shape[3])

                            lightmap_part = lightmap[:, :, start_row:end_row, start_col:end_col]
                            lightmap_reconstruct_part = lightmap_reconstruct[:, :, start_row:end_row, start_col:end_col]
                            mask_part = mask[start_row:end_row, start_col:end_col]
                            valid_mask = mask_part >= 127

                            if (np.any(valid_mask) and lightmap_part.max() != 0):
                                p_val = Utils.cal_psnr(lightmap_part, lightmap_reconstruct_part, mask_part)
                                s_val = Utils.cal_ssim(lightmap_part, lightmap_reconstruct_part)
                                l_val = Utils.cal_lpips(lightmap_part, lightmap_reconstruct_part)

                                # Add to local list
                                curr_psnr_list.append(p_val)
                                curr_ssim_list.append(s_val)
                                curr_lpips_list.append(l_val)

            # --- NEW: Print Sub-scores for this lightmap ---
            # Calculate means for this specific lightmap
            lm_mean_psnr = np.mean(curr_psnr_list) if curr_psnr_list else 0
            lm_mean_ssim = np.mean(curr_ssim_list) if curr_ssim_list else 0
            lm_mean_lpips = np.mean(curr_lpips_list) if curr_lpips_list else 0
            lm_total_time = np.sum(curr_time_list)

            print(f"[{idx + 1}/{total_maps}] {level}-{lm_id} | "
                  f"PSNR: {lm_mean_psnr:.2f} | "
                  f"SSIM: {lm_mean_ssim:.4f} | "
                  f"LPIPS: {lm_mean_lpips:.4f} | "
                  f"Time: {lm_total_time * 1000:.1f}ms", flush=True)

            # --- Add local stats to global stats ---
            psnr_list.extend(curr_psnr_list)
            ssim_list.extend(curr_ssim_list)
            lpips_list.extend(curr_lpips_list)
            time_list.extend(curr_time_list)

        # 获取模型大小，用于计算压缩率
        model_size = Utils.get_folder_size(PARAMETERS)

        psnr_score = ((np.clip(np.mean(psnr_list), psnr_worst_value, psnr_best_value) - psnr_worst_value) / (
                psnr_best_value - psnr_worst_value)) * 100
        ssim_score = ((np.clip(np.mean(ssim_list), ssim_worst_value, ssim_best_value) - ssim_worst_value) / (
                ssim_best_value - ssim_worst_value)) * 100
        lpips_score = ((lpips_worst_value - np.clip(np.mean(lpips_list), lpips_best_value, lpips_worst_value)) / (
                lpips_worst_value - lpips_best_value)) * 100
        time_worst_value *= TEST_K / 205
        time_score = (time_worst_value - (np.clip(np.sum(time_list), time_best_value, time_worst_value))) / (
                time_worst_value - time_best_value) * 100

        # Note: Compression Score will be inaccurate if running partial test
        # because model_size counts the whole folder, but data_size only counts the processed maps.
        compression_score = (compression_worst_value - np.clip(model_size / data_size, compression_best_value,
                                                               compression_worst_value)) / (
                                    compression_worst_value - compression_best_value) * 100
        weighted_score = 0.05 * psnr_score + 0.05 * ssim_score + 0.05 * lpips_score + 0.2 * compression_score + 0.15 * time_score

        print(f"\n================ FINAL RESULTS ================")
        print(f"tgac&yinqin&final:{weighted_score:.4f}", flush=True)

        results = {
            "PSNR Score": psnr_score,
            "SSIM Score": ssim_score,
            "LPIPS Score": lpips_score,
            "Compression Ratio Score": compression_score,
            "Inference Time Score": time_score,
            "Score": weighted_score
        }

        print(json.dumps(results, indent=4), flush=True)

        with open("scores.json", "w") as f:
            json.dump(results, f)