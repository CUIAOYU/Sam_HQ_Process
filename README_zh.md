# SAM_HQ 图像批量处理与分析工具

如果你正在寻找一款能够批量处理图片、利用 SAM_HQ (Segment Anything in High Quality) 模型生成高质量图像掩码 (mask)，并且能按自定义规则筛选这些掩码，同时还能方便地保存、加载处理参数并监控系统资源的工具，那么这款基于 PyQt6 图形界面的小工具可能正好能帮到你。

## 它能做什么？

* **双功能选项卡**：在一个界面中集成了“批量处理”和“单图分析”两大功能模块。
* **批量处理模式**：
    * **智能文件遍历**：选择一个文件夹，它能找出里面所有的图片（包括子文件夹里的），一张张帮你处理。支持 `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` 格式。
    * **SAM_HQ 模型支持**：你可以加载自己下载的 SAM_HQ 预训练模型 (`.pth` 文件)，用它来全自动生成超高质量的图像掩码。
    * **掩码生成参数可调**：SAM_HQ 自动掩码生成器的那些参数，你都可以根据需要调整。
    * **两步筛选掩码**：
        * **第一步：看区域属性**：根据每个掩码区域的像素面积 (Area)、总聚合值 (Intensity)，以及这两者的比率 (Ratio = Intensity/Area) 来做初步筛选。
        * **第二步：看像素聚合值**：通过第一步的掩码，工具会进一步看区域里每个像素的聚合值（计算方式是 `100 - (灰度值 / 255.0) * 100.0`），以进行更精细的筛选。
    * **结果可视化与预览**：筛选通过的掩码轮廓，连同它们的统计信息，都会被画在原图上并保存。提供“查看最后输出”功能，方便快速检查。
    * **数据导出清晰**：所有筛选通过的掩码的结构化数据（图像名、掩码索引、面积、强度、比率）可以一键导出为 `.txt` 文件。
* **单图分析模式**：
    * **聚合值分布可视化**：选择单张图片，工具可以分析其所有像素的“聚合值”分布情况，并生成直方图。
    * **关键节点识别**：自动在分布图上识别并标注出波峰和波谷，帮助你理解图像内容的灰度分布特征。
    * **详细统计数据**：提供总像素、平均/最小/最大聚合值等关键统计信息。
    * **图表导出**：可以将生成的分析图表保存为 PNG, PDF 或 SVG 格式。
* **通用功能**：
    * **简单好用的图形界面**：基于 PyQt6 开发，点几下鼠标就能上手。
    * **参数配置管理**：可以一键**保存/加载**所有筛选参数和 SAM_HQ 模型参数到 JSON 文件。
    * **输出文件好管理**：输出文件会自动保存在根据当前参数命名的文件夹里，并能处理来自不同子文件夹的同名文件，避免覆盖。
    * **实时处理日志与进度**：界面上会显示详细的处理进度、总体进度条以及可能出现的任何问题。
    * **系统资源监控**：在界面上实时显示 CPU、内存、GPU 及显存的使用率（需安装相应库）。

## 运行环境准备

要跑这个工具，你的电脑上得有 Python 环境，还得装一些必要的库。**建议使用 Windows 10 或 Windows 11 操作系统。**
我们将主要使用 Anaconda Navigator 来创建和管理环境，然后使用 `pip` 命令来安装包。

**1. 先装 Anaconda (如果电脑上还没有)**

* 去 [Anaconda 官网](https://www.anaconda.com/products/distribution) 下载对应你操作系统的最新版 Anaconda Distribution，并完成安装。

**2. 使用 Anaconda Navigator 创建环境**

* 打开 Anaconda Navigator。
* 在左侧导航栏选择 “Environments”（环境）选项卡。
* 点击底部的 “Create”（创建）按钮。
* 在弹出的对话框中：
    * **Name (名称)**: 输入一个环境名称，例如 `sam_hq_env`。
    * **Packages (包)**: 确保 Python 被选中，并在其版本下拉菜单中选择 **3.10** 或 **3.11** (推荐使用与 PyTorch 兼容性较好的版本)。
* 点击 “Create” 创建环境。这可能需要一些时间。

**3. 安装 Git (如果电脑上还没有)**

* Git 用于从 GitHub 下载代码。如果你的电脑还没有安装 Git，请访问 [git-scm.com](https://git-scm.com/downloads) 下载并安装。

**4. 使用 `pip` 安装所需的 Python 包 (通过终端)**

* 在 Anaconda Navigator 的 “Environments” 选项卡中，确保你刚刚创建的 `sam_hq_env` 环境被选中。
* 点击环境名称旁边的播放按钮（▶），然后选择 “Open Terminal”（打开终端）。这会打开一个已经激活了 `sam_hq_env` 环境的命令行窗口。
* 在打开的终端中，使用以下 `pip install` 命令来安装各个包：
    * **安装 PyTorch (包含 torchvision 和 torchaudio):**
        * 访问 [PyTorch 官网](https://pytorch.org/get-started/locally/)。
        * 在页面上选好你的操作系统 (Windows)、包管理器 (`Pip`)、编程语言 (Python) 和 CUDA 版本（如果你的电脑有 NVIDIA 显卡并且想用 GPU 加速，就选对应的 CUDA 版本；例如 CUDA 11.8 或 CUDA 12.1）。
        * 复制官网提供的 `pip` 安装命令，并在终端中执行。例如，Windows + CUDA 11.8 的命令可能是：
            ```bash
            pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
            ```
            如果不需要 GPU 或没有 NVIDIA GPU，请选择 CPU 版本对应的命令。

    * **安装其他核心依赖包:**
        在同一个终端中，继续执行以下命令：
        ```bash
        pip install PyQt6 opencv-python numpy tifffile pycocotools timm matplotlib
        ```
        * **关于 `pycocotools` 的小提示**：在 Windows 上直接用 `pip install pycocotools` 容易碰到编译错误。如果真遇上了，可以试试 `pip install pycocotools-windows`。

    * **安装可选的系统监控依赖包 (推荐):**
        ```bash
        pip install psutil
        pip install nvidia-ml-py # 如果你有 NVIDIA 显卡并希望监控
        ```

**5. 克隆 SAM-HQ 仓库并进行本地安装 (需要使用终端)**

* 如果你上一步打开的终端还开着，并且已经安装完了所有包，可以直接在该终端中继续操作。否则，请重新在 Anaconda Navigator 中为 `sam_hq_env` 环境打开一个终端。
* 在打开的终端中，执行以下命令来克隆 SAM-HQ 仓库并进行本地安装：
    ```bash
    git clone [https://github.com/SysCV/sam-hq.git](https://github.com/SysCV/sam-hq.git)
    cd sam-hq
    pip install -e .
    cd ..
    ```
    这样操作后，它就会被安装成一个叫 `segment_anything` 的包。

**6. 下载 SAM_HQ 模型权重文件**

* 你需要从 SAM-HQ 项目（比如它的 GitHub 仓库）下载预训练好的模型权重文件。这些文件通常是 `.pth` 结尾的，比如 `sam_hq_vit_h.pth`、`sam_hq_vit_l.pth` 或者 `sam_hq_vit_b.pth`。把它们下载到你电脑上一个方便找得到的地方。

## 界面操作和参数都是啥意思？

程序主界面分为两个选项卡：“单图分析” 和 “批量处理”。

### 1. Single Image Analysis (单图分析)

这个选项卡用于深入分析单张图片的像素灰度分布。

* **Image Selection (图像选择)**: 点击 "Select Image" 选择一张图片。
* **Analyze Button (分析按钮)**: 点击 "Analyze Aggregation Distribution" 开始分析，右侧会生成聚合值分布的直方图。
* **Aggregation Distribution Chart (聚合分布图)**:
    * **X轴**: 聚合值 (0-100)，越高代表像素越暗。
    * **Y轴**: 像素数量。
    * 图上会自动标注出显著的波峰 (Peaks) 和波谷 (Valleys)。
* **Statistics (统计数据)**: 显示总像素数、平均/最小/最大聚合值以及波峰/波谷的具体坐标。
* **Save/Clear Chart (保存/清除图表)**: 可以将分析图表保存为图片，或清除当前图表。

### 2. Batch Processing (批量处理)

这个选项卡用于对整个文件夹的图片进行自动化的掩码生成和筛选。

#### **File Selection (选文件)**

* **Input Folder (输入文件夹)**: 点 "Select Input Folder" 按钮，选一个装着原始图片的文件夹。
* **Model File (模型文件)**: 点 "Select SAM Model File (.pth)" 按钮，选你下载好的 SAM_HQ 模型权重文件。

#### **Configuration (参数配置)**

* **Save Config (保存配置)**: 保存当前所有筛选和 SAM 参数到 JSON 文件。
* **Load Config (加载配置)**: 从 JSON 文件加载并应用之前保存的参数。

#### **Phase 1: Region Property Filters (第一步筛选：看区域属性)**

* **Min Area (A) Threshold (最小面积阈值)**: 一个区域至少要有多少像素才算数。默认：`0`。
* **Min Intensity (I) Threshold (最小总聚合值阈值)**: 一个区域里所有像素的聚合值加起来的总和，不能低于这个数。默认：`0.0`。
* **Min Ratio (R=I/A) Threshold (最小比率阈值)**: 总聚合值 (Intensity) 除以面积 (Area) 得到的比率，不能低于这个数。默认：`0.0`。
* **Enable Phase 1 Filters (启用第一步筛选)**:
    * **Area (A)**: 勾上才用面积筛选。
    * **Intensity/Ratio (I/R)**: 勾上才用总聚合值和比率筛选。

#### **Phase 2: Pixel Aggregate Filters (第二步筛选：看像素聚合值)**

* **Min Aggregate Threshold (最小聚合值阈值)**: 像素的聚合值（范围 0-100）不能低于这个数。默认：`0`。
* **Max Aggregate Threshold (最大聚合值阈值)**: 像素的聚合值（范围 0-100）不能高于这个数。默认：`100`。
    * **筛选逻辑**: 只有聚合值在设定范围内的像素才被用于最终的面积、总聚合值和比率的计算。但是，只要一个区域里**存在至少一个**这样的有效像素，程序就会将这个区域**原始的、完整的轮廓**绘制出来并保留。

#### **SAM_HQ Auto Mask Generator Parameters (SAM_HQ 自动找图参数)**

* **Points Per Side (每条边上撒多少点)**: 在图片上均匀撒点的密度。点越多，看得越细，但也越慢。默认：`32`。
* **Points Per Batch (每次处理多少点)**: 一次处理多少个点。默认：`64`。
* **Pred IoU Thresh (预测效果好坏的门槛)**: 预测掩码与真实情况重合度的阈值。默认：`0.88`。
* **Stability Score Thresh (找得稳不稳的门槛)**: 掩码稳定性分数的阈值。默认：`0.95`。
* **Stability Score Offset (稳定分微调)**: 稳定性分数的一个调整参数。默认：`1.0`。
* **Box NMS Thresh (去掉重复框框的门槛)**: 用于合并高度重叠的边界框。默认：`0.7`。
* **Crop N Layers (把大图切成几块看)**: 对大图进行分层裁剪处理的层数。0表示不裁剪。默认：`0`。
* **Crop NMS Thresh (切块后去重门槛)**: 用于合并在不同裁剪块中找到的重叠掩码。默认：`0.7`。
* **Crop Overlap Ratio (切块时边上留多少重叠)**: 相邻裁剪块之间的重叠比例。默认：`0.341` (即 512/1500)。
* **Crop Points Downscale Factor (切块后撒点密度调整)**: 在裁剪的小块上撒点的密度调整因子。默认：`1`。
* **Min Mask Region Area (最小保留区域面积)**: SAM在生成过程中会忽略掉小于此面积的区域。默认：`0`。
* **Output Mode (结果保存格式)**: 掩码的输出格式。默认：`binary_mask`。
* **Reset SAM_HQ Params to Defaults (恢复默认设置)**: 将上述所有 SAM 参数恢复到程序内置的默认值。

#### **Operations (操作按钮区)**

* **Start Processing All Images (开始处理所有图片)**: 输入和模型都选好后，点此开始批量处理。
* **Stop Processing (停止处理)**: 处理过程中随时可以安全地停止任务。
* **Open Output Folder (打开输出文件夹)**: 直接打开保存结果的文件夹。
* **Export All Values (导出全部数值)**: 处理完成后，将所有通过筛选的掩码数据导出为 `.txt` 文件。
* **View Last Output (查看最后输出)**: 使用系统默认程序打开最后一张处理好的图片。

#### **System Monitor & Processing Log (系统监控与处理日志)**

* 这两个区域分别实时显示系统资源占用情况和程序的详细运行日志。

## 一些小提醒

* 第一次加载模型，或者处理特别大的图片时，可能会花点时间，请耐心等待。
* 本工具处理的图像尺寸单边最大为 4096 像素。如果图像的长或宽超过此限制，该图像将被跳过处理。
* 确保你的电脑硬盘有足够的空间来保存处理完的图片和导出的数据文件。
* 如果 GPU 监控显示 "N/A" 或错误，请确保您的 NVIDIA 驱动程序是最新的，并且 `nvidia-ml-py` (提供 `pynvml`) 包已正确安装在当前 Python 环境中。
* 如果在 Windows 上安装 `pycocotools` 不顺利，优先试试 `pip install pycocotools-windows`。

## 如何贡献

非常欢迎各种形式的贡献！

* **提 Bug 或建议**: 发现问题或者有改进想法？请到 GitHub 的 [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) 页面提出来。描述越详细越好，比如怎么复现、截图、你的系统和软件版本等等。
* **贡献代码**: 想直接改代码？欢迎！请走标准的 GitHub Fork & Pull Request 流程。最好是先开个 Issue 讨论一下你想做的改动。

## 报告问题

遇到 Bug 或有功能需求，请直接在仓库的 [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) 页面开新的 Issue。
