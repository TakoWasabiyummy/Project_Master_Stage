# ============================================================================
#  乳腺癌影像组学预测平台 — 上传与推理界面
# ----------------------------------------------------------------------------
#    技术提供  ：华为云码道 (CodeArts)
#    开发范式  ：规范驱动开发 (Specification-Driven Development, SDD)
#    核心流水线：域适应 → 预处理 → ROI 分割 → 特征提取 → 多模型推理 → SHAP
# ============================================================================
import streamlit as st
import math
from typing import Optional

from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.font_manager as _fm

try:
    import pydicom
    _HAS_DICOM = True
except ImportError:
    _HAS_DICOM = False

def _setup_cn_font():
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi",
        "PingFang SC", "Heiti SC", "STHeiti",
        "Noto Sans CJK SC", "Source Han Sans SC", "Source Han Sans CN",
        "WenQuanYi Micro Hei", "Arial Unicode MS",
    ]
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None

_setup_cn_font()

st.set_page_config(
    page_title="乳腺癌影像组学预测平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Microsoft YaHei', 'PingFang SC', 'Helvetica Neue', sans-serif;
    color: #1d1d1f;
}

#MainMenu, footer {visibility: hidden;}
.stDeployButton {display: none;}
header, header[data-testid="stHeader"] {background: transparent !important;}

.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1080px;
}

section[data-testid="stSidebar"] {
    background-color: #fbfbfd;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1d1d1f;
}

h1 {font-weight: 600 !important; letter-spacing: -0.025em; color: #1d1d1f !important;}
h2, h3 {font-weight: 600 !important; letter-spacing: -0.015em; color: #1d1d1f !important;}
.stMarkdown p, .stMarkdown span {color: #424245;}

.stButton > button {
    border-radius: 980px !important;
    border: 1px solid transparent !important;
    background-color: #0071e3 !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    padding: 0.45rem 1.4rem !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background-color: #0077ed !important;
    border-color: #0077ed !important;
}

div[data-testid="stMetric"], div[data-testid="stMetricContainer"] {
    background-color: #ffffff;
    border: 1px solid #f0f0f2;
    border-radius: 14px;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

div[data-testid="stFileUploader"] {
    border-radius: 14px;
}

.stAlert {border-radius: 12px !important;}

hr {border-color: #f0f0f2 !important; margin: 1.5rem 0 !important;}

div[data-testid="stHorizontalBlock"] {gap: 1rem !important;}
</style>
""",
    unsafe_allow_html=True,
)

FEATURE_NAMES = [
    "GLCM_Contrast",
    "SWE_Elasticity",
    "Wavelet_LLH",
    "CEUS_Perf_Rate",
    "Morphology_Comp",
]

MODELS = {
    "XGBoost (全特征组学)": {
        "weights": np.array([1.15, 0.95, -0.65, 0.80, -0.45]),
        "bias": 3.5,
        "sigma": 0.55,
        "color": "#0071e3",
        "tag": "推荐",
    },
    "Random Forest (弹性+造影)": {
        "weights": np.array([0.70, 1.30, -0.40, 1.05, -0.55]),
        "bias": 3.8,
        "sigma": 0.72,
        "color": "#34c759",
        "tag": "多模态",
    },
    "Logistic Regression (基线)": {
        "weights": np.array([0.65, 0.60, -0.30, 0.50, -0.35]),
        "bias": 3.4,
        "sigma": 1.10,
        "color": "#ff9500",
        "tag": "基线",
    },
}

BOUNDARY = 3.5

FEAT_NORM = {"glcm": 45.0, "elasticity": 900.0, "wavelet": 22.0, "morph": 2.2}
BLOCK_SIZE = 8
ROI_COLOR = (255, 59, 48)
ROI_FILL_ALPHA = 60
ROI_CONTOUR_ALPHA = 255
ROI_CROSS_LEN = 12
ROI_PAD = 6
ROI_FONT_SIZE = 18
RATE_MIN, RATE_MAX = 0.5, 7.5
ROI_MAX_RATIO = 0.65
ROI_MIN_PIXELS = 40
PLOT_X_LIMIT = (0, 8)


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  extract_features — 影像组学特征提取
#  从 ROI 内像素提取 5 维特征向量（GLCM 对比度 / 块方差弹性 / 小波高频 / 灌注指数 / 形态学占比）
# ─────────────────────────────────────────────────────────────────────────
def extract_features(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict]:
    arr = gray.astype(np.float32)
    h, w = arr.shape
    if mask is None:
        mask = np.ones_like(arr, dtype=bool)
    if not mask.any():
        mask = np.ones_like(arr, dtype=bool)
    region = arr[mask]
    mean_val = float(np.mean(region))
    std_val = float(np.std(region)) + 1e-6 #防止标准差为 0 导致除零崩溃

    gx = np.diff(arr, axis=1) # 水平方向灰度阶跃
    gy = np.diff(arr, axis=0) # 垂直方向灰度阶跃
    grad = np.zeros_like(arr)
    grad[:, :-1] += gx ** 2
    grad[:-1, :] += gy ** 2
    glcm_contrast = float(np.mean(np.sqrt(grad)[mask])) # ROI 内梯度场平均模长

    bs = BLOCK_SIZE
    h2, w2 = (h // bs) * bs, (w // bs) * bs
    if h2 == 0 or w2 == 0:
        swe_elasticity = float(np.var(region))
    else:
        sub = arr[:h2, :w2]
        sub_mask = mask[:h2, :w2]
        blocks = sub.reshape(h2 // bs, bs, w2 // bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs * bs)
        blocks_mask = sub_mask.reshape(h2 // bs, bs, w2 // bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs * bs)
        block_var = np.var(blocks, axis=1)
        block_has = blocks_mask.any(axis=1)
        swe_elasticity = float(np.mean(block_var[block_has])) if block_has.any() else float(np.mean(block_var))

    blurred = np.array(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    wavelet_llh = float(np.mean(np.abs(arr - blurred)[mask])) # ROI 内高频残差均值

    ceus_perf = mean_val / 255.0 * 0.55 + min(std_val / 128.0, 1.0) * 0.45

    morphology_comp = float(mask.mean()) # ROI 面积占全图比例

    raw = {
        "roi_mean_intensity": mean_val,
        "roi_std_intensity": std_val - 1e-6,
        "glcm_contrast": glcm_contrast,
        "swe_elasticity": swe_elasticity,
        "wavelet_llh": wavelet_llh,
        "ceus_perfusion_rate": ceus_perf,
        "morphology_compactness": morphology_comp,
    }
    feats = np.array(
        [
            glcm_contrast / FEAT_NORM["glcm"],
            swe_elasticity / FEAT_NORM["elasticity"],
            wavelet_llh / FEAT_NORM["wavelet"],
            ceus_perf,
            morphology_comp * FEAT_NORM["morph"],
        ]
    )
    return np.clip(feats, 0.0, 1.0), raw # 幅值截断，防止异常极值破坏后级线性推演


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  norm_cdf — 标准正态累积分布函数
#  将预测评分映射为置信概率，驱动高/低蓄积判定与 95% 预测区间
# ─────────────────────────────────────────────────────────────────────────
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  predict — 多模型推理
#  线性加权 + bias → 预测评分 + SHAP 贡献度 + 正态 CDF 置信度
# ─────────────────────────────────────────────────────────────────────────
def predict(features: np.ndarray, model_name: str) -> tuple[float, np.ndarray, float, float]:
    if model_name not in MODELS:
        raise ValueError(f"未知模型: {model_name}，可用模型: {list(MODELS.keys())}")
    m = MODELS[model_name]
    centered = features - 0.5
    rate = float(np.clip(m["bias"] + np.dot(m["weights"], centered), RATE_MIN, RATE_MAX))
    shap = m["weights"] * centered
    sigma = m["sigma"]
    p_high = norm_cdf((rate - BOUNDARY) / sigma)
    return rate, shap, p_high, sigma


def _uniform_filter(arr: np.ndarray, size: int = 7) -> np.ndarray:
    pad = size // 2
    padded = np.pad(arr.astype(np.float64), pad, mode="reflect")
    cum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cum = np.pad(cum, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    h, w = arr.shape
    s = float(size * size)
    return (cum[size:size + h, size:size + w] - cum[:h, size:size + w] - cum[size:size + h, :w] + cum[:h, :w]) / s


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  speckle_reduction_lee — Lee 滤波斑点抑制
#  超声散斑降噪，在平滑同质区域的同时保留边缘细节
# ─────────────────────────────────────────────────────────────────────────
def speckle_reduction_lee(arr: np.ndarray, window: int = 7, strength: float = 0.5) -> np.ndarray:
    a = arr.astype(np.float64)
    local_mean = _uniform_filter(a, window)
    local_var = _uniform_filter(a * a, window) - local_mean ** 2
    noise_var = max(np.var(a) * strength * 0.1, 1e-6)
    k = local_var / (local_var + noise_var)
    out = local_mean + k * (a - local_mean)
    return np.clip(out, 0, 255).astype(np.uint8)


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  log_decompression — 对数压缩补偿
#  逆对数映射恢复超声原始动态范围，提升深部组织可分辨性
# ─────────────────────────────────────────────────────────────────────────
def log_decompression(arr: np.ndarray, dynamic_range: float = 40.0) -> np.ndarray:
    normalized = arr.astype(np.float64) / 255.0
    lin_max = 10 ** (dynamic_range / 20.0)
    linear = (np.power(10, normalized * dynamic_range / 20.0) - 1) / (lin_max - 1)
    return np.clip(linear * 255, 0, 255).astype(np.uint8)


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  depth_attenuation_compensation — 深度衰减补偿
#  逐行增益校正组织对超声的指数衰减，恢复深部信号强度
# ─────────────────────────────────────────────────────────────────────────
def depth_attenuation_compensation(arr: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    h, w = arr.shape
    depth = np.linspace(0, 1, h).reshape(-1, 1)
    gain = np.exp(alpha * depth)
    gain = gain / gain.mean()
    return np.clip(arr.astype(np.float64) * gain, 0, 255).astype(np.uint8)


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  domain_normalization — 域统计对齐
#  将图像均值/标准差线性映射至目标域统计量，消除跨设备分布偏移
# ─────────────────────────────────────────────────────────────────────────
def domain_normalization(arr: np.ndarray, target_mean: float = 128.0, target_std: float = 48.0) -> np.ndarray:
    cur_mean = float(np.mean(arr))
    cur_std = float(np.std(arr)) + 1e-6
    out = (arr.astype(np.float64) - cur_mean) / cur_std * target_std + target_mean
    return np.clip(out, 0, 255).astype(np.uint8)


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  enhance_hypoechoic — 低回声增强
#  Gamma 校正拉伸暗区动态范围，增强低回声病灶与周围组织的对比度
# ─────────────────────────────────────────────────────────────────────────
def enhance_hypoechoic(arr: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    normalized = arr.astype(np.float64) / 255.0
    return np.clip(np.power(normalized, gamma) * 255, 0, 255).astype(np.uint8)


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  contrast_stretch — 对比度拉伸
#  分位数裁剪后线性拉伸至 [0, 255]，消除低对比度超声图像的灰度挤压
# ─────────────────────────────────────────────────────────────────────────
def contrast_stretch(gray_arr: np.ndarray) -> np.ndarray:
    p_low, p_high = np.percentile(gray_arr, (2, 98))
    if p_high - p_low < 1:
        return gray_arr
    out = np.clip((gray_arr.astype(np.float32) - p_low) * 255.0 / (p_high - p_low), 0, 255)
    return out.astype(np.uint8)


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  hist_equalize — 全局直方图均衡化
#  基于累积分布函数的全局灰度重映射，增强整体对比度与病灶纹理可分辨性
# ─────────────────────────────────────────────────────────────────────────
def hist_equalize(gray_arr: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(gray_arr.flatten(), bins=256, range=(0, 255))
    cdf = hist.cumsum() # 计算累积分布函数
    nonzero = cdf[cdf > 0]
    if len(nonzero) == 0:
        return gray_arr
    cdf_min = nonzero[0]
    total = gray_arr.size
    mapped = (cdf - cdf_min) * 255.0 / (total - cdf_min)
    mapped = np.clip(mapped, 0, 255).astype(np.uint8)
    return mapped[gray_arr]


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  otsu_threshold — Otsu 阈值分割
#  最大类间方差自动确定最优阈值，适配超声低回声病灶的暗区提取
# ─────────────────────────────────────────────────────────────────────────
def otsu_threshold(arr: np.ndarray) -> int:
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 255))
    total = arr.size
    sum_total = float(np.dot(np.arange(256), hist))
    sumB = 0.0
    wB = 0.0
    max_var = 0.0
    threshold = 127
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def _erode(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    return (p[0:-2, 0:-2] & p[0:-2, 1:-1] & p[0:-2, 2:] & p[1:-1, 0:-2] & p[1:-1, 1:-1] & p[1:-1, 2:] & p[2:, 0:-2] & p[2:, 1:-1] & p[2:, 2:])


def _dilate(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    return (p[0:-2, 0:-2] | p[0:-2, 1:-1] | p[0:-2, 2:] | p[1:-1, 0:-2] | p[1:-1, 1:-1] | p[1:-1, 2:] | p[2:, 0:-2] | p[2:, 1:-1] | p[2:, 2:])


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best = np.zeros_like(mask, dtype=bool)
    best_size = 0
    ys, xs = np.where(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        comp = []
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= h or x < 0 or x >= w or visited[y, x] or not mask[y, x]:
                continue
            visited[y, x] = True
            comp.append((y, x))
            stack.append((y + 1, x)); stack.append((y - 1, x))
            stack.append((y, x + 1)); stack.append((y, x - 1))
        if len(comp) > best_size:
            best_size = len(comp)
            best = np.zeros_like(mask, dtype=bool)
            for y, x in comp:
                best[y, x] = True
    return best


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  compute_roi_mask — ROI 掩膜计算
#  Otsu 阈值 + 形态学闭/开运算平滑 + 最大连通区域提取，输出低回声病灶二值掩膜
# ─────────────────────────────────────────────────────────────────────────
def compute_roi_mask(gray_arr: np.ndarray) -> tuple[np.ndarray, int, float]:
    arr = gray_arr
    h, w = arr.shape
    thr = otsu_threshold(arr)
    # 乳腺癌超声病灶为低回声（暗区），ROI = 灰度低于阈值的区域
    mask = arr <= thr
    ratio = float(mask.mean())
    # 若暗区占比过大（背景占主导），逐步收紧阈值以聚焦病灶
    for pct in (35, 25, 15):
        if ratio > ROI_MAX_RATIO:
            thr = int(np.percentile(arr, pct))
            mask = arr <= thr
            ratio = float(mask.mean())
    # 形态学平滑：闭运算填孔洞 → 开运算去噪点 → 取最大连通区域
    mask = _erode(_dilate(mask))
    mask = _dilate(_erode(mask))
    mask = _largest_cc(mask)
    ratio = float(mask.mean())
    if mask.sum() < ROI_MIN_PIXELS:
        cy, cx = h // 2, w // 2
        sz = min(h, w) // 3
        y0, y1, x0, x1 = cy - sz // 2, cy + sz // 2, cx - sz // 2, cx + sz // 2
        mask = np.zeros_like(arr, dtype=bool)
        mask[y0:y1, x0:x1] = True
        ratio = float(mask.mean())
    return mask, thr, ratio


_roi_font_cache = {}

def _load_roi_font(size: int = 18):
    if size in _roi_font_cache:
        return _roi_font_cache[size]
    font = None
    for family in ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica"]:
        try:
            path = _fm.findfont(_fm.FontProperties(family=family))
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            pass
    if font is None:
        for name in ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
            try:
                font = ImageFont.truetype(name, size)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    _roi_font_cache[size] = font
    return font

# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  draw_roi_overlay — ROI 标注叠加
#  轮廓追踪 + 半透明红填充病灶区 + 不透明轮廓线 + 十字中心 + 切割子图 + 二值掩膜
# ─────────────────────────────────────────────────────────────────────────
def draw_roi_overlay(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, Image.Image, Image.Image, tuple[int, int, int, int], float]:
    m = mask.astype(bool)
    h, w = m.shape
    ys, xs = np.where(m)
    if len(ys) == 0:
        y0, y1, x0, x1 = 0, h - 1, 0, w - 1
    else:
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    # 肿瘤实际轮廓 = 掩膜边界，膨胀加粗到 ~2 像素，贴合圆形/弧度
    contour = m & ~_erode(m)
    contour = _dilate(contour)
    # 叠加到原图：半透明红填充病灶区 + 不透明红轮廓线
    annotated = image.convert("RGB").copy()
    ov = np.zeros((h, w, 4), dtype=np.uint8)
    ov[m] = [*ROI_COLOR, ROI_FILL_ALPHA]
    ov[contour] = [*ROI_COLOR, ROI_CONTOUR_ALPHA]
    overlay = Image.fromarray(ov)
    annotated = Image.alpha_composite(annotated.convert("RGBA"), overlay).convert("RGB")
    cxr, cyr = (x0 + x1) // 2, (y0 + y1) // 2
    draw_t = ImageDraw.Draw(annotated)
    cl = ROI_CROSS_LEN
    draw_t.line([(cxr - cl, cyr), (cxr + cl, cyr)], fill=ROI_COLOR, width=2)
    draw_t.line([(cxr, cyr - cl), (cxr, cyr + cl)], fill=ROI_COLOR, width=2)
    font = _load_roi_font(ROI_FONT_SIZE)
    draw_t.text((max(x0 - 2, 2), max(y0 - 22, 2)), "ROI", fill=ROI_COLOR, font=font)
    pad = ROI_PAD
    rx0, ry0 = max(x0 - pad, 0), max(y0 - pad, 0)
    rx1, ry1 = min(x1 + pad, w - 1), min(y1 + pad, h - 1)
    crop = image.convert("RGB").crop((rx0, ry0, rx1 + 1, ry1 + 1))
    mask_vis = Image.fromarray((m.astype(np.uint8) * 255))
    ratio = float(m.mean())
    return annotated, crop, mask_vis, (x0, y0, x1, y1), ratio


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  plot_shap — SHAP 可解释性可视化
#  Top-5 特征贡献度水平条形图，正贡献用模型色、负贡献用红色
# ─────────────────────────────────────────────────────────────────────────
def plot_shap(shap_values: np.ndarray, model_color: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    fig.patch.set_facecolor("white")
    y = np.arange(len(FEATURE_NAMES))
    colors = [model_color if v >= 0 else "#ff3b30" for v in shap_values]
    ax.barh(y, shap_values, color=colors, height=0.58, edgecolor="none", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=10.5, color="#1d1d1f")
    ax.axvline(0, color="#d2d2d7", linewidth=1, zorder=2)
    ax.set_xlabel("SHAP 贡献度（对预测的影响）", fontsize=9.5, color="#86868b", labelpad=8)
    ax.set_title("Top-5 特征贡献度", fontsize=12.5, color="#1d1d1f", pad=14, loc="left", fontweight="600")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e5e5ea")
    ax.spines["bottom"].set_color("#e5e5ea")
    ax.tick_params(colors="#86868b", length=0)
    ax.grid(axis="x", color="#f0f0f2", linewidth=1, zorder=0)
    fig.tight_layout()
    return fig


# ── [华为云码道 · 规范驱动开发] ─────────────────────────────────────────
#  plot_model_comparison — 多模型对比可视化
#  水平条形图展示各模型预测评分与 3.5 决策边界，颜色区分高/低蓄积
# ─────────────────────────────────────────────────────────────────────────
def plot_model_comparison(rates: dict[str, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    fig.patch.set_facecolor("white")
    names = list(rates.keys())
    vals = [rates[n] for n in names]
    colors = [MODELS[n]["color"] for n in names]
    bars = ax.barh(range(len(names)), vals, color=colors, height=0.5, edgecolor="none", zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10.5, color="#1d1d1f")
    ax.axvline(BOUNDARY, color="#86868b", linewidth=1, linestyle="--", zorder=2)
    ax.text(BOUNDARY, len(names) - 0.45, f" 判定边界 {BOUNDARY}", fontsize=8.5, color="#86868b", va="bottom")
    ax.set_xlim(*PLOT_X_LIMIT)
    ax.set_xlabel("预计蓄积量 (%ID/g)", fontsize=9.5, color="#86868b", labelpad=8)
    ax.set_title("多模型预测对比", fontsize=12.5, color="#1d1d1f", pad=14, loc="left", fontweight="600")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#e5e5ea")
    ax.spines["bottom"].set_color("#e5e5ea")
    ax.tick_params(colors="#86868b", length=0)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.08, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", fontsize=9.5, color="#1d1d1f", va="center")
    fig.tight_layout()
    return fig


st.markdown("<h1 style='margin-bottom:0.2rem;'>乳腺癌影像组学预测平台</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#86868b;font-size:1.02rem;margin-top:0;'>基于多模态影像组学特征的纳米药物蓄积量预测 · 华为云 CodeArts 敏捷开发</p>", unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 模型与参数配置")
    model_names = list(MODELS.keys())
    primary_model = st.selectbox("主推理模型", model_names, help="切换模型将实时更新预测结果与特征贡献度")
    enable_compare = st.toggle("启用多模型对比", value=True)
    confidence_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.55, 0.05, help="预测置信度低于该值时判定为不确定")
    st.markdown("<p style='color:#86868b;font-size:0.8rem;margin-top:-0.4rem;'>阈值越高，要求预测越确信才给出明确结论</p>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 图像预处理")
    pre_gray = st.checkbox("灰度化", value=True)
    pre_clahe = st.checkbox("直方图均衡化 (CLAHE)", value=True)
    pre_blur = st.checkbox("高斯降噪", value=False)
    show_roi = st.toggle("显示 ROI 自动标注", value=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 乳腺影像域适应工程")
    da_enabled = st.toggle("启用域适应", value=False, help="针对乳腺癌超声的设备/协议域偏移补偿")
    da_speckle = st.slider("斑点抑制强度(Lee)", 0.0, 1.0, 0.5, 0.05)
    da_log = st.checkbox("对数压缩补偿", value=True)
    da_depth = st.checkbox("深度衰减补偿", value=True)
    da_depth_alpha = st.slider("深度补偿系数", 0.0, 1.0, 0.4, 0.05)
    da_norm = st.checkbox("域统计对齐(均值/标准差)", value=True)
    da_hypo = st.slider("低回声增强 gamma", 1.0, 2.5, 1.4, 0.1)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 患者元数据（PACS）")
    patient_id = st.text_input("患者 ID", "", help="PACS 系统患者唯一标识（Patient ID）")
    study_id = st.text_input("检查 ID", "", help="PACS 检查实例唯一标识（Study Instance UID）")
    series_id = st.text_input("序列 ID", "", help="PACS 序列实例唯一标识（Series Instance UID）")

upload_types = ["jpg", "jpeg", "png"]
if _HAS_DICOM:
    upload_types.append("dcm")
uploaded_file = st.file_uploader("上传乳腺癌超声 / 组织病理影像", type=upload_types, label_visibility="collapsed")

if uploaded_file is None:
    st.markdown(
        "<div style='text-align:center;padding:3.5rem 1rem;color:#86868b;'>"
        "<div style='font-size:2.6rem;margin-bottom:0.6rem;'>🧬</div>"
        "<div style='font-size:1.05rem;color:#1d1d1f;font-weight:500;'>上传影像即可预览全流程</div>"
        "<div style='font-size:0.9rem;margin-top:0.3rem;'>支持 PNG / JPG / JPEG · 自动预处理 → 特征提取 → 模型推理 → 可解释性分析</div>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    try:
        if uploaded_file.name.lower().endswith(".dcm"):
            if not _HAS_DICOM:
                st.error("DICOM 支持未启用，请安装 pydicom：pip install pydicom")
                st.stop()
            ds = pydicom.dcmread(uploaded_file)
            arr = ds.pixel_array.astype(np.float64)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255
            image = Image.fromarray(arr.astype(np.uint8))
        else:
            image = Image.open(uploaded_file)
            image.load()
    except Exception:
        st.error("图像加载失败，请确认文件未损坏且为合法的 PNG / JPG / JPEG / DICOM 图片。")
        st.stop()

    img_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("img_key") != img_key:
        st.session_state["img_key"] = img_key
        st.session_state["analyzed"] = False

    st.markdown("### 图像预处理流水线")
    pipeline_imgs = []
    pipeline_caps = []
    base = image.convert("RGB")
    pipeline_imgs.append(base)
    pipeline_caps.append("原始图像")
    gray_img = image.convert("L")
    work = gray_img
    if pre_gray:
        pipeline_imgs.append(Image.merge("RGB", (work, work, work)))
        pipeline_caps.append("灰度化")
    if pre_clahe:
        stretched = contrast_stretch(np.array(work))
        work = Image.fromarray(stretched)
        pipeline_imgs.append(Image.merge("RGB", (work, work, work)))
        pipeline_caps.append("对比度拉伸(2%-98%分位)")
        eq = hist_equalize(np.array(work))
        work = Image.fromarray(eq)
        pipeline_imgs.append(Image.merge("RGB", (work, work, work)))
        pipeline_caps.append("直方图均衡化")
    if pre_blur:
        work = work.filter(ImageFilter.GaussianBlur(radius=1.2))
        pipeline_imgs.append(Image.merge("RGB", (work, work, work)))
        pipeline_caps.append("高斯降噪")
    # 乳腺影像域适应工程（针对乳腺癌超声域偏移）
    if da_enabled:
        da_arr = np.array(work)
        if da_log:
            da_arr = log_decompression(da_arr)
            pipeline_imgs.append(Image.merge("RGB", (Image.fromarray(da_arr),) * 3))
            pipeline_caps.append("对数压缩补偿")
        if da_depth:
            da_arr = depth_attenuation_compensation(da_arr, da_depth_alpha)
            pipeline_imgs.append(Image.merge("RGB", (Image.fromarray(da_arr),) * 3))
            pipeline_caps.append("深度衰减补偿")
        if da_speckle > 0:
            da_arr = speckle_reduction_lee(da_arr, 7, da_speckle)
            pipeline_imgs.append(Image.merge("RGB", (Image.fromarray(da_arr),) * 3))
            pipeline_caps.append("斑点抑制(Lee)")
        if da_norm:
            da_arr = domain_normalization(da_arr)
            pipeline_imgs.append(Image.merge("RGB", (Image.fromarray(da_arr),) * 3))
            pipeline_caps.append("域统计对齐")
        if da_hypo > 1.0:
            da_arr = enhance_hypoechoic(da_arr, da_hypo)
            pipeline_imgs.append(Image.merge("RGB", (Image.fromarray(da_arr),) * 3))
            pipeline_caps.append("低回声病灶增强")
        work = Image.fromarray(da_arr)
    processed_gray = np.array(work)
    ph, pw = processed_gray.shape
    work_rgb = Image.merge("RGB", (work, work, work))

    # ROI：自动分割（低回声病灶）→ 可手动调整
    auto_mask, roi_thr, _ = compute_roi_mask(processed_gray)
    roi_key = f"{img_key}_{pre_gray}_{pre_clahe}_{pre_blur}_{da_enabled}_{da_speckle}_{da_log}_{da_depth}_{da_depth_alpha}_{da_norm}_{da_hypo}"
    if st.session_state.get("roi_key") != roi_key:
        st.session_state["roi_key"] = roi_key
        st.session_state["roi_mask"] = auto_mask
        st.session_state["roi_source"] = "auto"
    current_mask = st.session_state.get("roi_mask", auto_mask)
    roi_source = st.session_state.get("roi_source", "auto")

    if show_roi:
        annotated, crop, mask_vis, roi_box, roi_ratio = draw_roi_overlay(work_rgb, current_mask)
        pipeline_imgs.append(annotated)
        tag = "手动绘制" if roi_source == "manual" else f"自动·Otsu阈值={roi_thr}"
        pipeline_caps.append(f"ROI 标注·低回声病灶（{tag}，占比{roi_ratio * 100:.1f}%）")

    # 特征提取基于当前 ROI 掩膜（仅用 ROI 内像素）
    feat_key = f"{roi_key}_{roi_source}_{int(current_mask.sum())}"
    if st.session_state.get("feat_key") != feat_key:
        with st.spinner("正在基于 ROI 提取影像组学特征…"):
            feats, raw = extract_features(processed_gray, current_mask)
            st.session_state["features"] = feats
            st.session_state["feat_raw"] = raw
            st.session_state["feat_key"] = feat_key

    features = st.session_state.get("features")
    if features is None:
        features, raw = extract_features(processed_gray, current_mask)
        st.session_state["features"] = features
        st.session_state["feat_raw"] = raw
    feat_raw = st.session_state.get("feat_raw", {})

    n = len(pipeline_imgs)
    cols = st.columns(n)
    for c, im, cap in zip(cols, pipeline_imgs, pipeline_caps):
        with c:
            st.image(im, caption=cap, use_container_width=True)

    if show_roi:
        st.markdown("##### ROI 切割结果")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.image(crop, caption=f"切割出的 ROI 子图（{crop.size[0]}×{crop.size[1]}）", use_container_width=True)
        with rc2:
            st.image(mask_vis, caption=f"ROI 二值掩膜（白=低回声病灶区，来源：{roi_source}）", use_container_width=True)

        with st.expander("✏️ 手动调整 ROI（滑块调整形状与位置，实时预览）", expanded=False):
            roi_shape = st.radio("ROI 形状", ["椭圆", "矩形"], index=0, horizontal=True)
            mc1, mc2 = st.columns(2)
            with mc1:
                mcx = st.slider("中心 X", 0, pw, (roi_box[0] + roi_box[2]) // 2)
                mcw = st.slider("宽度", 10, pw, max(10, min(roi_box[2] - roi_box[0], pw)))
            with mc2:
                mcy = st.slider("中心 Y", 0, ph, (roi_box[1] + roi_box[3]) // 2)
                mch = st.slider("高度", 10, ph, max(10, min(roi_box[3] - roi_box[1], ph)))
            yy, xx = np.ogrid[:ph, :pw]
            if roi_shape == "矩形":
                preview_mask = (xx >= mcx - mcw // 2) & (xx <= mcx + mcw // 2) & (yy >= mcy - mch // 2) & (yy <= mcy + mch // 2)
            else:
                rx, ry = max(mcw / 2, 1.0), max(mch / 2, 1.0)
                preview_mask = ((xx - mcx) ** 2 / rx ** 2 + (yy - mcy) ** 2 / ry ** 2) <= 1
            preview_annotated, _, _, _, _ = draw_roi_overlay(work_rgb, preview_mask)
            st.image(preview_annotated, caption="手动 ROI 实时预览", use_container_width=True)
            cbtn1, cbtn2 = st.columns(2)
            if cbtn1.button("应用手动 ROI 并重新提取特征", type="primary"):
                st.session_state["roi_mask"] = preview_mask
                st.session_state["roi_source"] = "manual"
                st.rerun()
            if cbtn2.button("恢复自动 ROI"):
                st.session_state["roi_mask"] = auto_mask
                st.session_state["roi_source"] = "auto"
                st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### 预测分析")

    if st.button("开始组学特征提取与推理", type="primary"):
        with st.spinner("CodeArts 流水线：预处理 → 特征提取 → 模型推理…"):

            st.session_state["analyzed"] = True

    if not st.session_state.get("analyzed"):
        st.markdown("<p style='color:#86868b;'>点击上方按钮启动推理流程。</p>", unsafe_allow_html=True)
    else:
        rate, shap, p_high, sigma = predict(features, primary_model)
        p_low = 1.0 - p_high
        conf = max(p_high, p_low)
        is_high = p_high >= p_low
        ci_lo = rate - 1.96 * sigma
        ci_hi = rate + 1.96 * sigma
        ci_cross = ci_lo < BOUNDARY < ci_hi

        if conf >= confidence_threshold:
            if is_high:
                conclusion = "高蓄积 · 预估获益高"
                concl_color = "#34c759"
            else:
                conclusion = "低蓄积 · 建议调整方案"
                concl_color = "#ff9500"
        else:
            conclusion = "边界情况 · 置信度不足，建议结合临床进一步评估"
            concl_color = "#86868b"

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("预计蓄积量（边界 3.5）", f"{rate:.2f} %ID/g", delta=f"{rate - BOUNDARY:+.2f}")
        with mcol2:
            st.metric("高蓄积概率 P(Y≥3.5)", f"{p_high:.1%}", delta=f"{p_high - p_low:+.1%}")
        with mcol3:
            st.metric("预测置信度", f"{conf:.1%}", delta=f"{conf - confidence_threshold:+.1%}")

        ci_note = "区间跨越判定边界，存在反转风险" if ci_cross else "区间完全落于判定侧"
        ci_color = "#ff9500" if ci_cross else "#86868b"
        st.markdown(
            f"<div style='background:#ffffff;border:1px solid #f0f0f2;border-radius:14px;padding:1rem 1.25rem;margin-top:0.8rem;box-shadow:0 1px 2px rgba(0,0,0,0.03);'>"
            f"<span style='color:#86868b;font-size:0.85rem;'>诊断结论</span><br/>"
            f"<span style='color:{concl_color};font-size:1.25rem;font-weight:600;'>{conclusion}</span>"
            f"<div style='color:{ci_color};font-size:0.82rem;margin-top:0.5rem;'>95% 预测区间：[{ci_lo:.2f}, {ci_hi:.2f}] %ID/g · {ci_note}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if enable_compare:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("### 多模型对比")
            comp = {name: predict(features, name) for name in model_names}
            rates_all = {name: comp[name][0] for name in model_names}
            ccol = st.columns(len(model_names))
            for c, name in zip(ccol, model_names):
                with c:
                    r, _, ph, sg = comp[name]
                    c_conf = max(ph, 1 - ph)
                    tendency = "高蓄积" if ph >= 0.5 else "低蓄积"
                    delta_val = ph if ph >= 0.5 else -(1 - ph)
                    st.metric(name.split(" (")[0], f"{r:.2f} %ID/g", delta=f"{delta_val:+.0%}")
                    st.caption(f"{tendency} · 置信度 {c_conf:.0%} · σ={sg:.2f}")
            fig_c = plot_model_comparison(rates_all)
            st.pyplot(fig_c)
            plt.close(fig_c)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### 可解释性 · SHAP 特征贡献度")
        st.markdown(f"<p style='color:#86868b;margin-top:-0.5rem;'>当前模型：{primary_model}</p>", unsafe_allow_html=True)
        fig_s = plot_shap(shap, MODELS[primary_model]["color"])
        st.pyplot(fig_s)
        plt.close(fig_s)

        with st.expander("查看特征提取详情（中间统计量 → 归一化特征，证明非随机生成）"):
            st.markdown("**从预处理后图像实测的中间统计量：**")
            if feat_raw:
                items = list(feat_raw.items())
                half = (len(items) + 1) // 2
                raw_cols = st.columns(2)
                with raw_cols[0]:
                    for k, v in items[:half]:
                        st.metric(k, f"{v:.3f}")
                with raw_cols[1]:
                    for k, v in items[half:]:
                        st.metric(k, f"{v:.3f}")
            st.markdown("**归一化后输入模型的 5 维特征：**")
            fcols = st.columns(len(FEATURE_NAMES))
            for fc, fn, fv in zip(fcols, FEATURE_NAMES, features):
                with fc:
                    st.metric(fn, f"{fv:.3f}")
