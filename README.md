**LightMap（光照贴图）压缩与重建**

### 环境准备

1. 安装基础环境

   - 使用 conda 创建 Python 3.11 环境，CUDA 12.9

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 安装 tiny-cuda-nn（如需要）

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

4. 数据集位置
   - `/HPRC_Test1/Data/`保存在`/trunk/`底下

### 任务目标

需要实现一个 LightMap 压缩与重建系统，评估指标：

- PSNR（目标：≥55dB）
- SSIM（目标：≥0.999）
- LPIPS（目标：≤0.0003）
- 压缩率（目标：越小越好）
- 推理时间（目标：越快越好）

最终得分权重：

- PSNR: 5%
- SSIM: 5%
- LPIPS: 5%
- 压缩率: 20%
- 推理时间: 15%

### 数据结构

- 光照数据文件：`lightmapRawData_*`，RGB 三通道，float32
- 掩码文件：`lightmapCoverage_*`，标识有效/无效像素（-1=无效，127=有效）
- 时间范围：0-24 小时，包含 1-24 整点时刻，以及 5.9 和 18.1 两个特殊时刻

### 开发流程

1. 示例代码

   ```bash
   python ExampleTrain.py  # 训练示例模型
   python Test.py          # 测试接口
   ```

   - 结果保存在`scores.json`里

2. 研究示例实现

   - 查看 `ExampleModel.py` 了解模型结构（使用 tiny-cuda-nn 的 HashGrid）
   - 查看 `ExampleTrain.py` 了解训练流程
   - 查看 `Interface.py` 了解接口实现方式

3. 设计算法

   - 主要改进 `Examplemodel` 和 `Exampletrain`
   - 改进模型架构（提高质量、降低压缩率、加快推理）
   - 优化训练策略

4. 保存参数
   - 所有模型参数保存在 `Parameters/` 文件夹， 运行`ExampleTrain`可得到
   - 必须是原始二进制文件（不能额外压缩）
   - 参考 `ExampleTrain.py` 的保存方式
