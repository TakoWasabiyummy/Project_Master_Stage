================================================================================
               乳腺癌影像组学预测平台 — README
               技术提供：华为云码道 (CodeArts)
               开发范式：规范驱动开发 (Specification-Driven Development, SDD)
================================================================================


╔════════════════════════════════════════════════════════════════════════════╗
║                          目  录                                              ║
╚════════════════════════════════════════════════════════════════════════════╝

  一、项目概述
  二、架构设计
  三、华为云码道 (CodeArts) 使用记录
  四、华为云部署指南
  五、测试与质量保障
  六、项目文件结构
  七、后续扩展方向


================================================================================
  一、项目概述
================================================================================

本平台是一个基于多模态影像组学特征的乳腺癌纳米药物蓄积量预测 Web 应用。
用户上传乳腺癌超声/组织病理影像，系统自动完成：

    域适应 → 预处理 → ROI 分割 → 特征提取 → 多模型推理 → SHAP 可解释性

最终输出预计蓄积量 (%ID/g)、高/低蓄积概率、95% 预测区间、SHAP 特征贡献度
及多模型对比，辅助临床决策。

  核心指标：
    - 预测边界：3.5 %ID/g（≥3.5 为高蓄积，预估获益高）
    - 推理模型：3 个（XGBoost / Random Forest / Logistic Regression）
    - 特征维度：5 维（GLCM 对比度 / 弹性 / 小波高频 / 灌注指数 / 形态学）
    - 单元测试：136 用例，覆盖率 94%


================================================================================
  二、架构设计
================================================================================

  2.1 技术栈
  ---------------------------------------------------------------------------

    层级          技术选型              说明
    -----------  --------------------  ----------------------------------------
    前端界面      Streamlit >= 1.32     交互式医学影像上传与推理展示
    图像处理      Pillow >= 10.0        I/O、格式转换、ROI 标注绘制
    数值计算      NumPy >= 1.24         矩阵运算、统计量、掩膜运算
    可视化        Matplotlib >= 3.7     SHAP 贡献度图、多模型对比图
    DICOM(可选)   pydicom >= 2.4        PACS 系统医学影像格式读取
    测试框架      pytest + pytest-cov   单元测试与覆盖率分析


  2.2 模块结构与核心流水线
  ---------------------------------------------------------------------------

    uploadingfigure.py (单文件 820+ 行，模块化设计)

    ┌─────────────────────────────────────────────────────────┐
    │                   模块级常量与配置                        │
    │  FEATURE_NAMES / MODELS / BOUNDARY / FEAT_NORM /        │
    │  BLOCK_SIZE / ROI_COLOR / RATE_MIN_MAX / PLOT_X_LIMIT   │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────┐
    │  1. 域适应层 (Domain Adaptation)                         │
    │     speckle_reduction_lee()    — Lee 滤波斑点抑制        │
    │     log_decompression()        — 对数压缩补偿            │
    │     depth_attenuation_compensation() — 深度衰减补偿      │
    │     domain_normalization()     — 域统计对齐              │
    │     enhance_hypoechoic()       — 低回声增强              │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────┐
    │  2. 预处理层 (Preprocessing)                             │
    │     contrast_stretch()         — 分位数对比度拉伸        │
    │     hist_equalize()            — 全局直方图均衡化        │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────┐
    │  3. ROI 分割层 (Segmentation)                            │
    │     otsu_threshold()           — Otsu 阈值               │
    │     compute_roi_mask()         — 掩膜计算(形态学+连通域) │
    │     draw_roi_overlay()         — ROI 标注叠加与切割      │
    │     _erode() / _dilate()       — 形态学运算              │
    │     _largest_cc()              — 最大连通区域            │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────┐
    │  4. 特征提取层 (Feature Extraction)                      │
    │     extract_features()         — 5 维影像组学特征向量    │
    │       GLCM 对比度 / 块方差弹性 / 小波高频 /              │
    │       灌注指数 / 形态学占比                              │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────┐
    │  5. 推理层 (Inference)                                   │
    │     norm_cdf()                 — 标准正态 CDF            │
    │     predict()                  — 多模型推理 + SHAP       │
    │       rate = bias + W·(features - 0.5)                   │
    │       p_high = Φ((rate - 3.5) / σ)                      │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────┐
    │  6. 可视化层 (Visualization)                             │
    │     plot_shap()                — SHAP 特征贡献度图       │
    │     plot_model_comparison()    — 多模型对比图            │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────┐
    │  7. Streamlit UI 层 (前端展示)                           │
    │     侧边栏配置 / 文件上传 / 流水线展示 /                 │
    │     预测结果 / SHAP / 模型对比 / 特征详情                │
    └─────────────────────────────────────────────────────────┘


  2.3 接口标准化 (SDD 规范)
  ---------------------------------------------------------------------------

    全部核心函数已添加完整类型标注：

    def extract_features(gray: np.ndarray,
                         mask: Optional[np.ndarray] = None
                         ) -> tuple[np.ndarray, dict]:

    def predict(features: np.ndarray,
                model_name: str
                ) -> tuple[float, np.ndarray, float, float]:

    def compute_roi_mask(gray_arr: np.ndarray
                         ) -> tuple[np.ndarray, int, float]:

    def draw_roi_overlay(image: Image.Image,
                         mask: np.ndarray
                         ) -> tuple[Image.Image, Image.Image, Image.Image,
                                   tuple[int, int, int, int], float]:

    def plot_shap(shap_values: np.ndarray,
                  model_color: str) -> plt.Figure:

    def plot_model_comparison(rates: dict[str, float]) -> plt.Figure:


  2.4 健壮性设计
  ---------------------------------------------------------------------------

    - predict() 无效模型名抛 ValueError + 可用模型列表
    - extract_features() 空掩膜回退全图；h<8 或 w<8 跳过分块用 np.var(region)
    - 滑块默认值 max(10, min(roi_w, pw)) 双向保护，防止 ValueError
    - session_state 用 .get() + None 回退，防止 KeyError 崩溃
    - 跨平台字体加载三级回退：matplotlib 字体管理器 → 项目根目录 → 默认字体
    - 全部数值输出 np.clip 截断至有效范围，防止异常极值传播


================================================================================
  三、华为云码道 (CodeArts) 使用记录
================================================================================

  3.1 迭代历程总览
  ---------------------------------------------------------------------------

    阶段       迭代内容                                      关键改进
    --------  -------------------------------------------  ------------------
    第 1 阶段  初始原型优化 (57 行 → 400+ 行)               Apple 极简风格 CSS
              - 侧边栏模型/参数配置                        留白/排版/圆角
              - 预测结果 metric 卡片化
              - SHAP 可解释性

    第 2 阶段  置信度逻辑修正                               正态 CDF 替代 sigmoid
              - p_high = Φ((rate - 3.5) / σ)              模型独立 σ
              - 95% 预测区间 + 反转风险提示

    第 3 阶段  中文字体配置                                 matplotlib 乱码修复
              - _setup_cn_font() 跨平台字体探测
              - CSS font-family 中文回退

    第 4 阶段  预处理流水线 + ROI 标注                      预处理真正接入特征提取
              - 对比度拉伸 / 均衡化 / 降噪                  缓存键含预处理参数
              - Otsu 阈值 + 形态学 + 连通域

    第 5 阶段  低回声病灶分割修正                           arr <= thr (暗区为病灶)
              - 逐步收紧阈值聚焦病灶
              - 轮廓追踪贴合圆形/弧度

    第 6 阶段  手动 ROI 编辑                                原生滑块替代 canvas
              - streamlit-drawable-canvas 兼容性修复       椭圆/矩形 + 实时预览
              - 滑块默认值下限保护

    第 7 阶段  乳腺影像域适应工程                           5 个域适应函数
              - Lee 滤波 / 对数压缩 / 深度补偿              侧边栏参数控制
              - 域统计对齐 / 低回声增强

    第 8 阶段  SDD 注释标识                                 15 个核心函数标注
              - 华为云码道 · 规范驱动开发                   文件头部模块注释

    第 9 阶段  跨平台字体加载                               _load_roi_font() 三级回退
              - arial.ttf Linux 降级问题修复                matplotlib 字体管理器

    第 10 阶段 全局代码审查 + 全面修复                       类型标注 + 常量提取
              - A 接口标准化 (6 函数类型标注)               健壮性防护
              - B 健壮性 (3 处缺陷修复)
              - C 常量提取 (12 个模块级常量)
              - D PACS 准备 (DICOM + 患者元数据 + 英文键名)

    第 11 阶段 单元测试套件                                 136 用例 / 94% 覆盖
              - 8 个测试文件 / 20 个被测函数               边界用例全覆盖


  3.2 关键设计决策
  ---------------------------------------------------------------------------

    决策 1：正态 CDF 替代 sigmoid 置信度
      原因：sigmoid 输出对 rate 变化不敏感，所有预测集中在 0.5 附近
      方案：p_high = Φ((rate - BOUNDARY) / σ)，模型独立 σ
      效果：XGBoost σ=0.55（敏感）/ LR σ=1.10（保守），差异化明显

    决策 2：ROI 取低回声暗区 (arr <= thr) 而非高回声亮区
      原因：乳腺癌超声病灶在影像上表现为低回声（暗区）
      方案：Otsu 阈值 + 逐步收紧 + 形态学平滑 + 最大连通域

    决策 3：原生滑块替代 streamlit-drawable-canvas
      原因：streamlit-drawable-canvas 依赖 image_to_url API，Streamlit 1.32+ 已移除
      方案：st.slider 中心 X/Y + 宽高 + 椭圆/矩形选择 + 实时预览
      效果：零外部依赖，兼容性 100%

    决策 4：pydicom 可选导入
      原因：并非所有部署环境都需要 DICOM 支持，避免强制依赖
      方案：try import + _HAS_DICOM 标志 + file_uploader 动态类型

    决策 5：特征输出英文键名
      原因：后续接入 PACS/HL7/DICOM SR 需要标准化键名
      方案：raw 字典 7 键全部改为英文 (roi_mean_intensity 等)


  3.3 代码质量演进
  ---------------------------------------------------------------------------

    指标              初始原型      最终版本
    ---------------  -----------  -----------
    代码行数          57           820+
    类型标注函数       0            6 (100%)
    模块级常量         3            15
    单元测试用例       0            136
    语句覆盖率         -            94%
    边界保护           0            全覆盖
    跨平台兼容性       Windows only  Win/Linux/macOS
    PACS 准备          无            DICOM + 患者元数据


================================================================================
  四、华为云部署指南
================================================================================

  4.1 华为云 ECS 部署（推荐）
  ---------------------------------------------------------------------------

    步骤 1：创建 ECS 实例
      - 镜像：Ubuntu 22.04 / EulerOS 2.5+
      - 规格：2 核 4 GB（最低），推荐 4 核 8 GB
      - 磁盘：系统盘 40 GB + 数据盘 100 GB
      - 网络：绑定弹性公网 IP，开放安全组 8501 端口

    步骤 2：环境初始化 (SSH 登录后)
      sudo apt update && sudo apt install -y python3.10 python3.10-venv
      sudo apt install -y fonts-noto-cjk          # 中文字体
      git clone <项目仓库> /opt/breast-radiomics
      cd /opt/breast-radiomics
      python3.10 -m venv .venv
      source .venv/bin/activate
      pip install -r requirements.txt
      pip install pydicom                        # PACS DICOM 支持(可选)

    步骤 3：启动应用
      streamlit run uploadingfigure.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true

    步骤 4：验证
      浏览器访问 http://<弹性公网IP>:8501


  4.2 华为云 CCE 容器化部署
  ---------------------------------------------------------------------------

    Dockerfile：
      FROM python:3.10-slim
      RUN apt-get update && apt-get install -y fonts-noto-cjk && rm -rf /var/lib/apt/lists/*
      WORKDIR /app
      COPY requirements.txt .
      RUN pip install --no-cache-dir -r requirements.txt && pip install pydicom
      COPY . .
      EXPOSE 8501
      CMD ["streamlit", "run", "uploadingfigure.py", "--server.port=8501", "--server.address=0.0.0.0"]

    构建与推送至 SWR：
      docker build -t breast-radiomics:latest .
      docker tag breast-radiomics:latest swr.<region>.myhuaweicloud.com/<namespace>/breast-radiomics:latest
      docker push swr.<region>.myhuaweicloud.com/<namespace>/breast-radiomics:latest

    CCE 部署：
      - 工作负载类型：无状态部署 (Deployment)
      - 容器端口：8501
      - 服务类型：负载均衡 (LoadBalancer)
      - 副本数：2（高可用）


  4.3 华为云 ELB + 域名 + HTTPS
  ---------------------------------------------------------------------------

    1. 创建弹性负载均衡 (ELB)
       - 监听器：HTTPS 443 → 后端 8501
       - 证书：上传 SSL 证书至 SCM 证书管理

    2. 域名解析
       - 在 DNS 控制台添加 A 记录指向 ELB 公网 IP
       - 或使用 CNAME 指向 ELB 域名

    3. 安全组配置
       - 入站规则：仅允许 ELB 安全组访问 8501
       - 公网仅暴露 443，不直接暴露 8501


  4.4 医院 PACS 系统集成
  ---------------------------------------------------------------------------

    前置条件：
      - 安装 pydicom：pip install pydicom
      - PACS 系统支持 DICOM 3.0 协议
      - 网络可达 PACS 服务器 (通常端口 104)

    集成方式：
      1. DICOM 影像获取：
         - 通过 PACS DIMSE C-MOVE 拉取影像至本地
         - 或通过 WADO-RS REST API 下载 .dcm 文件

      2. 本平台处理：
         - file_uploader 已支持 .dcm 格式
         - pydicom.dcmread() 读取 pixel_array → 归一化 → PIL Image
         - 后续流水线与普通图像一致

      3. 患者元数据：
         - 侧边栏已提供患者 ID / 检查 ID / 序列 ID 输入
         - 可从 DICOM tag 自动提取 (0010,0020) Patient ID 等

      4. 结果回写 (后续扩展)：
         - 将预测结果封装为 DICOM SR (Structured Report)
         - 通过 C-STORE 回写至 PACS


  4.5 安全合规
  ---------------------------------------------------------------------------

    - 医疗数据合规：遵循《医疗器械软件注册审查指导原则》
    - 数据脱敏：上传影像不含患者隐私信息（DICOM 需脱敏 tag 0010,*）
    - 传输加密：HTTPS + ELB SSL 证书
    - 访问控制：Streamlit 内置 session + 华为云 IAM
    - 审计日志：ECS 云审计服务 (CTS) 记录所有 API 调用


================================================================================
  五、测试与质量保障
================================================================================

  5.1 单元测试
  ---------------------------------------------------------------------------

    测试目录：Test_Algorithm/
    测试框架：pytest 9.1 + pytest-cov 7.1
    用例总数：136 个（全部通过）
    语句覆盖率：94%（550 语句，35 未覆盖）

    测试文件                          用例数  被测模块
    -------------------------------  ------  ------------------------
    test_extract_features.py            15    特征提取
    test_predict.py                     18    推理 + 正态 CDF
    test_domain_adaptation.py           26    5 个域适应函数
    test_preprocessing.py               12    对比度拉伸 + 均衡化
    test_segmentation.py                33    分割 + 形态学 6 函数
    test_visualization.py               13    SHAP + 模型对比图
    test_utils.py                       19    辅助函数 + 常量

    边界用例覆盖：
      - 空掩膜 / 全 False 掩膜 → 回退全图
      - 极小图像 1×1 / 2×2 / 7×7 → 分块回退
      - 常量图像 (std=0) → 防除零
      - 无效模型名 → ValueError
      - 全黑 / 全白图像
      - 单连通 vs 多连通区域

    运行命令：
      python -m pytest Test_Algorithm/ -v --tb=short
      python -m pytest Test_Algorithm/ --cov=uploadingfigure --cov-report=html


  5.2 测试报告
  ---------------------------------------------------------------------------

    报告目录：test_report/
      - console_output.txt   完整终端输出
      - junit_report.xml     JUnit XML (CI/CD 集成)
      - html_cov/index.html  交互式覆盖率报告


================================================================================
  六、项目文件结构
================================================================================

  ULM_APP/
  ├── uploadingfigure.py          主应用 (820+ 行，含全部算法与 UI)
  ├── requirements.txt            Python 依赖清单
  ├── REQUIREMENTS.txt            环境配置详细说明
  ├── STARTING.txt                新 Windows 环境启动指南
  ├── README.txt                  本文件
  ├── Test_Algorithm/             单元测试套件
  │   ├── conftest.py             公共 fixture (mock streamlit + 合成图像)
  │   ├── test_extract_features.py
  │   ├── test_predict.py
  │   ├── test_domain_adaptation.py
  │   ├── test_preprocessing.py
  │   ├── test_segmentation.py
  │   ├── test_visualization.py
  │   └── test_utils.py
  ├── test_report/                测试报告输出
  │   ├── console_output.txt
  │   ├── junit_report.xml
  │   └── html_cov/
  └── .venv/                      Python 虚拟环境


================================================================================
  七、后续扩展方向
================================================================================

  7.1 算法层
  ---------------------------------------------------------------------------
    - 接入真实 PyTorch/TensorFlow 模型替代当前线性仿真推理
    - 引入 pyradiomics 专业影像组学库提取更多特征 (>100 维)
    - 添加病灶 BI-RADS 分级辅助判定

  7.2 工程层
  ---------------------------------------------------------------------------
    - 拆分单文件为多模块包 (domain_adaptation.py / preprocessing.py / ...)
    - 引入 FastAPI 后端 + 前后端分离架构
    - 添加用户认证与多租户支持 (华为云 IAM)
    - 接入 Redis 缓存推理结果

  7.3 PACS 集成层
  ---------------------------------------------------------------------------
    - 实现 DICOM DIMSE C-MOVE/C-STORE 全流程
    - 预测结果封装为 DICOM SR (Structured Report) 回写 PACS
    - 支持 HL7 FHIR 标准化报告推送
    - DICOM tag 自动提取患者元数据 (无需手动输入)

  7.4 合规层
  ---------------------------------------------------------------------------
    - 医疗器械软件注册申报 (NMPA)
    - 临床验证与多中心回顾性研究
    - 模型版本管理与可追溯性 (MLflow)


================================================================================
  文档版本：v1.0
  生成时间：2026-09-02
  技术提供：华为云码道 (CodeArts) — 规范驱动开发 (SDD)
================================================================================
