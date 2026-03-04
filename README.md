# Grain Size Analysis System

基于传统图像处理的金相图晶粒自动分割与尺寸分析系统，符合 ASTM E112-13 标准。

## 功能

- 自动晶粒分割（Otsu 阈值 + Watershed 分水岭）
- 面积法（Planimetric/Jeffries）ASTM G 值计算
- 截线法（Heyn Intercept）ASTM G 值计算
- 异常晶粒识别（尺寸比法 / 长尾分布法 / 3σ 准则）
- 可视化标注图输出（分割图、分析图、异常图、分布直方图）

---

## 环境配置

```bash
micromamba env create -f environment.yml
micromamba activate grain-analysis
```

---

## 快速运行

### 单张图像

```bash
python main.py \
  --input /Users/siyuliu/Desktop/OM/steel_grain_size_dataset/RG/RG36_2_1.jpg \
  --pixels-per-micron 2.25
```

输出写入 `./data/RG36_2_1/`。

### 整个文件夹（批量）

```bash
python main.py \
  --input /Users/siyuliu/Desktop/OM/steel_grain_size_dataset/RG/ \
  --pixels-per-micron 2.25 \
  --output ./data
```

### 指定输出目录

```bash
python main.py \
  --input /path/to/image.jpg \
  --output /path/to/output \
  --pixels-per-micron 2.25
```

---

## CLI 参数说明

```
python main.py [OPTIONS]
```

### 必填参数

| 参数 | 缩写 | 说明 |
|---|---|---|
| `--input PATH` | `-i` | 输入图像文件或文件夹路径 |

### 通用参数

| 参数 | 缩写 | 默认值 | 说明 |
|---|---|---|---|
| `--output DIR` | `-o` | `./data` | 输出根目录 |
| `--pixels-per-micron FLOAT` | `-p` | `2.25` | 分辨率（像素/微米），500X 数据集为 2.25 |

### 预处理参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--gaussian-sigma FLOAT` | `3.0` | 高斯滤波标准差，控制去噪强度 |
| `--median-kernel INT` | `3` | 中值滤波核大小（奇数），去椒盐噪声 |
| `--clahe-clip FLOAT` | `2.0` | CLAHE 对比度限制，增强晶界可见性 |

### 分割参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--min-distance INT` | `50` | Watershed marker 最小间距（像素），控制晶粒最小尺寸 |
| `--closing-disk INT` | `2` | 形态学闭运算核半径，填补断裂晶界 |
| `--min-grain-area INT` | 自动 | 最小晶粒面积（像素²），不设则按分辨率自动估算 |
| `--remove-border` | 否 | 移除接触图像边缘的晶粒 |
| `--keep-border` | 是（默认）| 保留接触图像边缘的晶粒 |

### 截线法参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--n-lines-h INT` | `5` | 水平测试线数量 |
| `--n-lines-v INT` | `5` | 垂直测试线数量 |

### 异常检测参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--rule-a-threshold FLOAT` | `3.0` | 规则 A：d_max/d_avg 超过此值判定为异常 |
| `--rule-b-top-pct FLOAT` | `5.0` | 规则 B：检测前 X% 大晶粒的面积占比 |
| `--rule-b-area-frac FLOAT` | `0.30` | 规则 B：前 X% 晶粒面积占比超过此值判定为异常 |

---

## 输出结构

```
data/{image_name}/
├── {name}_original.png          # 原始输入图像
├── {name}_segmented.png         # 分割结果（伪彩色晶粒叠加）
├── {name}_area_method.png       # 面积法标注图（测量区域 + 统计）
├── {name}_intercept_method.png  # 截线法标注图（网格线 + 交点）
├── {name}_anomaly.png           # 异常晶粒高亮图（红色标注）
├── {name}_distribution.png      # 晶粒尺寸分布直方图
└── {name}_results.json          # 完整分析结果（JSON）
```

### results.json 关键字段

```json
{
  "grain_statistics": {
    "count": 142,
    "mean_diameter_um": 12.4,
    "std_diameter_um": 3.2
  },
  "area_method": {
    "astm_g_value": 3.4
  },
  "intercept_method": {
    "astm_g_value": 3.6
  },
  "anomaly_detection": {
    "has_anomaly": true,
    "total_anomalous_grains": 2
  }
}
```

---

## 代码结构

```
grain-analysis/
├── main.py              # CLI 入口（click）
└── src/
    ├── preprocessing.py # 灰度化、高斯滤波、中值滤波、CLAHE
    ├── segmentation.py  # Otsu 阈值 + Watershed 分水岭分割
    ├── analysis.py      # regionprops 特征提取、面积法、截线法
    ├── anomaly.py       # 三规则异常晶粒判定
    ├── visualization.py # 标注图生成（matplotlib）
    ├── io_utils.py      # 图像读取、目录管理、JSON 存储
    └── pipeline.py      # 端到端流程编排
```

---

## 测试数据集

`steel_grain_size_dataset/RG/`：480 张不锈钢 316L 光学显微镜图像
- 格式：JPG，400×300 px
- 分辨率：2.25 px/μm（500X）
- 参考分割标注：`steel_grain_size_dataset/RGMask/`

---

## 参考标准

- ASTM E112-13: 金属平均晶粒度测定方法
- GBT/6394-2017: 金属平均晶粒度测定方法
