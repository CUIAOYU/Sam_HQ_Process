# SAM_HQ 图像批量处理

如果你正在寻找一款能够批量处理图片、利用 SAM_HQ (Segment Anything in High Quality) 模型生成高质量图像掩码 (mask)，并且能按自定义规则筛选这些掩码的工具，那么这款基于 PyQt6 图形界面的小工具可能正好能帮到你。

## 它能做什么？

* **简单好用的图形界面**：基于 PyQt6 开发，点几下鼠标就能上手。
* **图片批量处理**：选择一个文件夹，它能找出里面所有的图片（包括子文件夹里的），一张张帮你处理。支持 `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`格式。
* **SAM_HQ 模型支持**：你可以加载自己下载的 SAM_HQ 预训练模型 (`.pth` 文件)，用它来全自动生成超高质量的图像掩码。
* **掩码生成参数可调**：SAM_HQ 自动掩码生成器的那些参数，你都可以根据需要调整。
* **两步筛选掩码**：
    * **第一步：看区域属性**：根据每个掩码区域的像素面积 (Area)、总亮度或聚合值 (Intensity)，以及这两者的比率 (Ratio = Intensity/Area) 来做初步筛选。这些筛选可以单独开关，阈值也能自己定。
    * **第二步：看像素聚合值**：通过第一步的掩码，工具会进一步看区域里每个像素的聚合值（计算方式是 `100 - (灰度值 / 255.0) * 100.0`），只保留那些聚合值在指定范围内的像素。
* **结果可视化**：筛选通过的掩码轮廓，连同它们的面积、总聚合值、比率这些信息，都会被画在原图上，然后保存成新的图片。
* **输出文件好管理**：
    * 处理好的图片会保存在你选的输入文件夹的同级目录，并放在一个根据当前参数命名的子文件夹里。
    * 如果图片来自子文件夹，输出文件名会自动加上相对路径信息（路径分隔符会变成下划线），这样就算不同子文件夹里有同名图片，输出时也不会互相覆盖。
* **实时处理日志**：界面上会显示详细的处理进度和可能出现的任何问题。
* **推荐操作系统**：本工具主要在 Windows 10 和 Windows 11环境下开发和测试，推荐在这些系统上运行以获得最佳体验。

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
    * **Packages (包)**: 确保 Python 被选中，并在其版本下拉菜单中选择 **3.11**。
* 点击 “Create” 创建环境。这可能需要一些时间。

**3. 安装 Git (如果电脑上还没有)**

* Git 用于从 GitHub 下载代码。如果你的电脑还没有安装 Git，请访问 [git-scm.com](https://git-scm.com/downloads) 下载并安装。

**4. 使用 `pip` 安装所需的 Python 包 (通过终端)**

* 在 Anaconda Navigator 的 “Environments” 选项卡中，确保你刚刚创建的 `sam_hq_env` 环境被选中。
* 点击环境名称旁边的播放按钮（▶），然后选择 “Open Terminal”（打开终端）。这会打开一个已经激活了 `sam_hq_env` 环境的命令行窗口。
* 在打开的终端中，使用以下 `pip install` 命令来安装各个包：

    * **安装 PyTorch (包含 torchvision 和 torchaudio):**
        * 访问 [PyTorch 官网](https://pytorch.org/get-started/locally/)。
        * 在页面上选好你的操作系统 (Windows)、包管理器 (`Pip`)、编程语言 (Python) 和 CUDA 版本（如果你的电脑有 NVIDIA 显卡并且想用 GPU 加速，就选对应的 CUDA 版本；CUDA 12.1 是个不错的选择）。
        * 复制官网提供的 `pip` 安装命令，并在终端中执行。例如，Windows + CUDA 12.1 的命令通常是：
            ```bash
            pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
            ```
            如果不需要 GPU 或没有 NVIDIA GPU，请选择 CPU 版本对应的命令。

    * **安装其他依赖包:**
        在同一个终端中，继续执行以下命令：
        ```bash
        pip install PyQt6 opencv-python numpy tifffile pycocotools timm
        ```
        * **关于 `pycocotools` 的小提示**：在 Windows 上直接用 `pip install pycocotools` 容易碰到编译错误。如果真遇上了：
            * 可以试试 `pip install pycocotools-windows` (一个社区维护的包，但不确定是否一直有效)。
            * 更稳妥的办法是去 [PyPI 官网的 pycocotools 页面](https://pypi.org/project/pycocotools/#files) 找找有没有已经编译好的 wheel 文件 (`.whl`)，要找文件名里包含你的 Python 版本（比如 `cp311` 代表 Python 3.11）和系统架构（比如 `win_amd64` 代表 Windows 64位）的。下载下来后，用 `pip install 你下载的文件路径.whl` 来安装。
            * 如果都不行，那可能就得装个 Microsoft C++ Build Tools 才能从源码编译了。

**5. 克隆 SAM-HQ 仓库并进行本地安装 (需要使用终端)**

* 如果你上一步打开的终端还开着，并且已经安装完了所有包，可以直接在该终端中继续操作。否则，请重新在 Anaconda Navigator 中为 `sam_hq_env` 环境打开一个终端。
* 在打开的终端中，执行以下命令来克隆 SAM-HQ 仓库并进行本地安装：
    ```bash
    git clone [https://github.com/SysCV/sam-hq.git](https://github.com/SysCV/sam-hq.git)
    cd sam-hq
    pip install -e .
    cd ..
    ```
    这样操作后，它就会被安装成一个叫 `segment_anything` 的包。请记下你克隆 `sam-hq` 仓库的完整路径 (例如 `C:\Users\你的用户名\sam-hq` 或你选择的其他路径)，后面会用到。

**6. 下载 SAM_HQ 模型权重文件**

* 你需要从 SAM-HQ 项目（比如它的 GitHub 仓库）下载预训练好的模型权重文件。这些文件通常是 `.pth` 结尾的，比如 `sam_hq_vit_h.pth`、`sam_hq_vit_l.pth` 或者 `sam_hq_vit_b.pth`。把它们下载到你电脑上一个方便找得到的地方。

## 界面操作和参数都是啥意思？

### 1. File Selection (选文件)

* **Input Folder (输入文件夹)**: 点 "Select Input Folder" 按钮，选一个装着原始图片的文件夹。程序会自动找这个文件夹和它里面所有子文件夹里的图片（支持 `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` 这些格式）。
* **Model File (模型文件)**: 点 "Select SAM Model File (.pth)" 按钮，选你之前下载好的 SAM_HQ 模型权重文件。

### 2. Phase 1: Region Property Filters (第一步筛选：看区域属性)

SAM_HQ 会生成很多初始的掩码区域，这里的筛选器就是针对这些区域的。

* **Min Area (A) Threshold (最小面积阈值)**: 一个区域至少要有多少像素才算数。默认：199。
* **Min Intensity (I) Threshold (最小总聚合值阈值)**: 一个区域里所有像素的聚合值（算法是 `100 - (灰度值/255)*100`）加起来的总和，不能低于这个数。默认：0.0。
* **Min Ratio (R=I/A) Threshold (最小比率阈值)**: 上面那个总聚合值 (Intensity) 除以面积 (Area) 得到的比率，不能低于这个数。默认：44.0。
* **Enable Phase 1 Filters (启用第一步筛选)**:
    * **Area (A)**: 勾上才用面积筛选。
    * **Intensity/Ratio (I/R)**: 勾上才用总聚合值和比率筛选。

### 3. Phase 2: Pixel Aggregate Filters (第二步筛选：看像素聚合值)

通过了第一步筛选的掩码区域，工具还会细看区域里的每个像素。

* **Min Aggregate Threshold (最小聚合值阈值)**: 像素的聚合值（范围 0-100）不能低于这个数。默认：35。
* **Max Aggregate Threshold (最大聚合值阈值)**: 像素的聚合值（范围 0-100）不能高于这个数。默认：100。
    （只有聚合值在这个设定范围内的像素才被认为是有效的，最终的面积、总聚合值、比率计算和画图也只用这些有效像素。）

### 4. SAM_HQ Auto Mask Generator Parameters (SAM_HQ 自动找图参数)
这部分是告诉电脑怎么更聪明、更细致地在图片上自动找出各种东西（我们叫它“生成掩码”）。你可以把这些参数想象成调节一台超级相机的各种旋钮和按钮，调好了就能拍出效果更好的“分割图”。

* **Points Per Side (每条边上撒多少点)**: 想象一下，电脑为了理解图片，会在图片上均匀地撒很多“观察点”。这个参数就是说，在图片每一条边上要撒多少个点。点越多，看得越细，但也越慢。默认是 32 个点。
* **Points Per Batch (每次处理多少点)**: 电脑处理这些“观察点”的时候，不是一个一个来，而是一批一批地处理。这个参数就是说，每一批处理多少个点。默认是 64 个点。
* **Pred IoU Thresh (预测效果好坏的门槛)**: 电脑找到一个东西后，会给它打个分，看看这个东西找得准不准。这个分数叫 IoU。这个参数就是设一个门槛，只有分数高于这个门槛的，才认为是找得不错的。默认是 0.86 分（满分是 1 分）。
* **Stability Score Thresh (找得稳不稳的门槛)**: 有时候电脑找东西会“手抖”，同一个东西可能一会儿看像这个，一会儿看像那个。这个参数也是设一个门槛，看电脑找这个东西稳不稳定，越稳定越好。默认是 0.92 分。
* **Stability Score Offset (稳定分微调)**: 这是上面那个“稳定分”的一个小调整参数，一般不用太管它。默认是 1.0。
* **Box NMS Thresh (去掉重复框框的门槛)**: 电脑找东西的时候，可能会对同一个东西画好几个框框。这个参数就是告诉电脑，如果好几个框框都圈着差不多的东西，就把多余的去掉，只留一个最好的。默认是 0.7。
* **Crop N Layers (把大图切成几块看)**: 如果图片特别大，电脑可能会把它切成几小块，一块一块地看，看得更仔细。这个参数就是说要切成几层来看。默认是 1 层（就是不怎么切或者只切一次）。
* **Crop NMS Thresh (切块后去重门槛)**: 上面说到切块看，那不同小块里可能也找到了重复的东西。这个参数跟“去掉重复框框的门槛”类似，也是用来去掉这些重复的。默认是 0.7。
* **Crop Overlap Ratio (切块时边上留多少重叠)**: 把图片切成小块时，为了不漏掉边缘的东西，相邻的小块之间会有一点点重叠。这个参数就是说重叠部分占多大比例。默认是 0.341 (差不多三分之一)。
* **Crop Points Downscale Factor (切块后撒点密度调整)**: 在切成的小块图片上撒“观察点”时，可以调整撒点的密度。这个参数就是调整密度的，数字越大，撒的点越稀疏。默认是 2。
* **Min Mask Region Area (最小保留区域面积)**: 电脑找到的东西，如果太小了（比如只有几个像素点那么大），可能就没啥用。这个参数就是说，找到的东西至少要有这么大面积才保留下来。默认是 100 个像素点。
* **Output Mode (结果保存格式)**: 电脑找到东西后，会用一种方式把这些“找到的区域”记录下来。这里有几种不同的记录格式可选。一般用默认的 `binary_mask` (就像一张黑白图，白色的地方是找到的，黑色的地方是没找到的) 就行。
* **Reset SAM_HQ Params to Defaults (恢复默认设置)**: 如果你把上面的参数调乱了，不知道哪个好，点这个按钮，所有这些“找图参数”就都回到一开始设好的样子了。

### 5. Operations (操作按钮区)

* **Start Processing All Images (开始处理所有图片)**: 输入文件夹和模型文件都选好了之后，点这个就开始处理找到的所有图片。
* **Stop Processing (停止处理)**: 处理过程中随时可以点这个按钮来安全地停掉当前任务。
* **Open Output Folder (打开输出文件夹)**: 点这个能直接打开自动生成的那个输出文件夹。输出文件夹一般会建在输入文件夹的上一级目录，文件夹名会包含一些参数信息。

### 6. Processing Log (处理日志区)

* 这个区域会告诉你程序正在干嘛、处理到哪儿了，以及有没有什么警告或者错误。

## 一些小提醒

* 第一次加载模型，或者处理特别大的图片时，可能会花点时间，耐心等等。
* 确保你电脑硬盘有足够的空间来保存处理完的图片。
* 如果在 Windows 上装 `pycocotools` 不顺利，优先试试找预编译好的 wheel 文件。
* 正确设置 `PYTHONPATH` 非常重要，不然 Python 可能找不到你从本地仓库装的 `segment_anything` (SAM-HQ) 库。

## 如何贡献

非常欢迎各种形式的贡献！

* **提 Bug 或建议**: 发现问题或者有改进想法？请到 GitHub 的 [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) 页面提出来。描述越详细越好，比如怎么复现、截图、你的系统和软件版本等等。
* **贡献代码**: 想直接改代码？欢迎！请走标准的 GitHub Fork & Pull Request 流程。最好是先开个 Issue 讨论一下你想做的改动。

## 报告问题

遇到 Bug 或有功能需求，请直接在仓库的 [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) 页面开新的 Issue。
