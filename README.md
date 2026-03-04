# Grain Size Analysis System

基于传统图像处理的金相图晶粒自动分割与尺寸分析系统，符合 ASTM E112-13 标准。

## 功能

- 自动晶粒分割（Otsu + Watershed 分水岭）
- 面积法（Planimetric/Jeffries）ASTM G 值计算
- 截线法（Heyn Intercept）ASTM G 值计算
- 异常晶粒识别（尺寸比法 / 长尾分布法 / 3σ 准则）
- 可视化标注图输出（分割图、分析图、异常图、分布直方图）

## 环境配置

使用 [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) 管理依赖：

```bash
micromamba env create -f environment.yml
micromamba activate grain-analysis
```

## 使用方式

```bash
# 分析单张图像
python main.py --input /path/to/image.jpg --pixels-per-micron 2.25

# 批量分析文件夹
python main.py --input /path/to/folder/ --pixels-per-micron 2.25

# 指定输出目录
python main.py --input /path/to/image.jpg --output ./data --pixels-per-micron 2.25
```

## 输出结构

```
data/{image_name}/
├── {name}_original.png
├── {name}_segmented.png
├── {name}_area_method.png
├── {name}_intercept_method.png
├── {name}_anomaly.png
├── {name}_distribution.png
└── {name}_results.json
```

## 代码结构

```
grain-analysis/
├── main.py              # CLI 入口（click）
└── src/
    ├── preprocessing.py # 灰度化、高斯滤波、CLAHE
    ├── segmentation.py  # Otsu + Watershed 分割
    ├── analysis.py      # regionprops 特征提取、面积法、截线法
    ├── anomaly.py       # 三规则异常晶粒判定
    ├── visualization.py # 标注图生成
    ├── io_utils.py      # 目录创建、JSON 存储
    └── pipeline.py      # 端到端流程编排
```

## 测试数据集

使用 `steel_grain_size_dataset/RG/`（480 张不锈钢 316L 光学显微镜图像，500X，400×300px，2.25 px/μm）。

## 参考标准

- ASTM E112-13: 金属平均晶粒度测定方法
- GBT/6394-2017: 金属平均晶粒度测定方法
