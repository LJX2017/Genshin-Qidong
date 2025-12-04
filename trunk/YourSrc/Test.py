import json
import os
import time
import numpy as np
import torch
import yaml

import Interface
import Utils

TEST_LIGHT_MAP_COUNT = 1

if __name__ == '__main__': 
    # 所有大赛使用的数据�?
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
        # 读取数据集下的配置文�?
        dataset_path = f'/mnt/c/HPRC_Test1/Data/{data_set}'
        config_file = 'config.json'
        with open(os.path.join(dataset_path, config_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        # --- MODIFICATION START ---
        # Select only the first 10 lightmaps for testing
        # You can change 10 to any number you prefer
        target_lightmaps = data['lightmap_list'][:TEST_LIGHT_MAP_COUNT]
        print(f"Debugging Mode: Testing on {len(target_lightmaps)} lightmaps only.")
        # --- MODIFICATION END ---
        lightmap_count = data['lightmap_count']
        
        # 初始化整个数据集的指标list
        psnr_list = []
        ssim_list = []
        lpips_list = []
        time_list = []
        data_size = 0

        # 计算所有lightmap的指�?
        for lightmap_config in target_lightmaps:
            # 从配置文件中获取lightmap的id、lightmap路径、mask路径、分辨率
            lightmap_names = lightmap_config['lightmaps']
            mask_names = lightmap_config['masks']
            resolution = lightmap_config['resolution']

            # 计算数据大小�?通道�?4个时刻，每个通道4个字�?
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
                    time_list.append(time_end - time_start)

                    # 测试Random Access
                    random_coord = np.zeros((1, 3))
                    random_y = np.random.randint(0, resolution['height'])
                    random_x = np.random.randint(0, resolution['width'])
                    random_coord[0, 0] = random_y
                    random_coord[0, 1] = random_x
                    random_coord[0, 2] = current_time
                    random_lightmap_reconstruct = your_interface.random_test(random_coord)
                    if not torch.allclose(random_lightmap_reconstruct[0, :], lightmap_reconstruct[0, :, random_y, random_x], atol=1e-2):
                        print(f"Random Access Test Failed at {current_time} {random_y} {random_x} !!!")
                        exit()
                    
                    # 读取原始lightmap数据计算指标
                    lightmap_path = os.path.join(dataset_path, "Data", lightmap_names[str(int(current_time * 100))])
                    mask_path = os.path.join(dataset_path, "Data", mask_names[str(int(current_time * 100))])
                                    
                    lightmap_data = np.fromfile(lightmap_path, dtype=np.float32)
                    mask_data = np.fromfile(mask_path, dtype=np.int8)
                    lightmap = lightmap_data.reshape(resolution['height'], resolution['width'], 3)
                    mask = mask_data.reshape(resolution['height'], resolution['width'])

                    # �?56*256分辨率计算一次指标，最后取平均�?
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
                            lightmap_reconstruct_part = lightmap_reconstruct[:, :,start_row:end_row, start_col:end_col]
                            mask_part = mask[start_row:end_row, start_col:end_col]
                            valid_mask = mask_part >= 127

                            if (np.any(valid_mask) and lightmap_part.max() != 0):
                                psnr_list.append(Utils.cal_psnr(lightmap_part, lightmap_reconstruct_part, mask_part))
                                ssim_list.append(Utils.cal_ssim(lightmap_part, lightmap_reconstruct_part))
                                lpips_list.append(Utils.cal_lpips(lightmap_part, lightmap_reconstruct_part))

        # 获取模型大小，用于计算压缩率
        model_size = Utils.get_folder_size('./Parameters')

        psnr_score = ((np.clip(np.mean(psnr_list), psnr_worst_value, psnr_best_value) - psnr_worst_value) / (psnr_best_value - psnr_worst_value)) * 100 
        ssim_score = ((np.clip(np.mean(ssim_list), ssim_worst_value, ssim_best_value) - ssim_worst_value) / (ssim_best_value - ssim_worst_value)) * 100
        lpips_score = ((lpips_worst_value - np.clip(np.mean(lpips_list), lpips_best_value, lpips_worst_value)) / (lpips_worst_value - lpips_best_value)) * 100
        time_score = (time_worst_value - (np.clip(np.sum(time_list), time_best_value, time_worst_value))) / (time_worst_value - time_best_value) * 100
        compression_score = (compression_worst_value - np.clip(model_size / data_size, compression_best_value, compression_worst_value)) / (compression_worst_value - compression_best_value) * 100
        weighted_score = 0.05 * psnr_score + 0.05 * ssim_score + 0.05 * lpips_score + 0.2 * compression_score + 0.15 * time_score

        print(f"tgac&yinqin&final:{weighted_score}",flush=True)

        results = {
            "PSNR Score": psnr_score,
            "SSIM Score": ssim_score,
            "LPIPS Score": lpips_score,
            "Compression Ratio Score": compression_score,
            "Inference Time Score": time_score,
            "Score": weighted_score
        }

        with open("scores.json", "w") as f:
            json.dump(results, f)

