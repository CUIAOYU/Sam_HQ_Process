# SAM_HQ Batch Image Processing and Analysis Tool

If you're looking for a tool that can batch process images, generate high-quality masks using the SAM_HQ (Segment Anything in High Quality) model, filter those masks based on custom rules, and conveniently save/load parameters while monitoring system resources, then this handy PyQt6-based GUI tool might be exactly what you need.

## What Can It Do?

* **Dual-Function Tabs**: Integrates "Batch Processing" and "Single Image Analysis" modules into one clean interface.
* **Batch Processing Mode**:
    * **Smart File Traversal**: Select a folder, and the tool will find and process all images within it (including subfolders). Supports `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff` formats.
    * **SAM_HQ Model Support**: Load your own pre-trained SAM_HQ models (`.pth` files) to automatically generate superior-quality image masks.
    * **Adjustable Mask Generation Parameters**: All parameters for the SAM_HQ automatic mask generator can be fine-tuned to fit your needs.
    * **Two-Phase Mask Filtering**:
        * **Phase 1: Region Properties**: Perform an initial filter based on the pixel area (Area), total intensity (Intensity), and the ratio of these two (Ratio = Intensity/Area) for each mask region.
        * **Phase 2: Pixel Aggregation Value**: For masks that pass Phase 1, the tool further examines the aggregation value of each pixel within the region (calculated as `100 - (grayscale_value / 255.0) * 100.0`) for more refined filtering.
    * **Result Visualization and Preview**: The contours of masks that pass the filters, along with their statistical information, are drawn on the original image and saved. A "View Last Output" feature allows for quick inspection.
    * **Clean Data Export**: Structured data for all filtered masks (image name, mask index, area, intensity, ratio) can be exported to a `.txt` file with a single click.
* **Single Image Analysis Mode**:
    * **Aggregation Value Distribution Visualization**: Select a single image, and the tool can analyze the "aggregation value" distribution of all its pixels and generate a histogram.
    * **Key Node Identification**: Automatically identifies and annotates peaks and valleys on the distribution chart to help you understand the image's grayscale distribution characteristics.
    * **Detailed Statistics**: Provides key stats like total pixels, average/min/max aggregation values.
    * **Chart Export**: The generated analysis chart can be saved as a PNG, PDF, or SVG file.
* **General Features**:
    * **User-Friendly GUI**: Developed with PyQt6 for an intuitive, point-and-click experience.
    * **Configuration Management**: **Save/Load** all filter and SAM_HQ parameters to/from a JSON file with one click.
    * **Organized Output**: Output files are automatically saved in a folder named according to the current parameters, and the tool handles same-named files from different subfolders to prevent overwrites.
    * **Real-time Logs and Progress**: The interface displays a detailed processing log, an overall progress bar, and any potential issues that arise.
    * **System Resource Monitoring**: Shows real-time CPU, memory, GPU, and VRAM usage directly in the interface (requires installing optional libraries).

## Environment Setup

To run this tool, you need a Python environment and several required libraries. **Windows 10 or Windows 11 is recommended.** We will primarily use Anaconda Navigator to create and manage the environment and `pip` to install packages.

**1. Install Anaconda (if you haven't already)**

* Go to the [official Anaconda website](https://www.anaconda.com/products/distribution) to download the latest Anaconda Distribution for your operating system and complete the installation.

**2. Create an Environment with Anaconda Navigator**

* Open Anaconda Navigator.
* Select the "Environments" tab on the left navigation bar.
* Click the "Create" button at the bottom.
* In the dialog box that appears:
    * **Name**: Enter an environment name, e.g., `sam_hq_env`.
    * **Packages**: Ensure Python is selected and choose version **3.10** or **3.11** from its dropdown menu (a version compatible with PyTorch is recommended).
* Click "Create" to build the environment. This may take some time.

**3. Install Git (if you haven't already)**

* Git is used to download code from GitHub. If you don't have Git installed, visit [git-scm.com](https://git-scm.com/downloads) to download and install it.

**4. Install Required Python Packages via Terminal**

* In Anaconda Navigator's "Environments" tab, make sure your newly created `sam_hq_env` environment is selected.
* Click the play button (▶) next to the environment name and select "Open Terminal". This will open a command-line window with the `sam_hq_env` environment already activated.
* In the terminal, use the following `pip install` commands:
    * **Install PyTorch (including torchvision and torchaudio):**
        * Visit the [official PyTorch website](https://pytorch.org/get-started/locally/).
        * On the page, select your OS (Windows), package manager (`Pip`), programming language (Python), and CUDA version (if you have an NVIDIA GPU and want to use it for acceleration, choose the appropriate version, e.g., CUDA 11.8 or CUDA 12.1).
        * Copy the `pip` installation command provided by the website and execute it in your terminal. For example, the command for Windows + CUDA 11.8 might be:
            ```bash
            pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
            ```
            If you don't need GPU support or don't have an NVIDIA GPU, select the CPU version command instead.

    * **Install Other Core Dependencies:**
        In the same terminal, continue by executing the following command:
        ```bash
        pip install PyQt6 opencv-python numpy tifffile pycocotools timm matplotlib
        ```
        * **A small tip for `pycocotools`**: A direct `pip install pycocotools` on Windows can sometimes lead to compilation errors. If you encounter this, try `pip install pycocotools-windows`.

    * **Install Optional System Monitoring Dependencies (Recommended):**
        ```bash
        pip install psutil
        pip install nvidia-ml-py # If you have an NVIDIA GPU and want to monitor it
        ```

**5. Clone and Locally Install the SAM-HQ Repository**

* If your terminal from the previous step is still open and all packages are installed, you can continue there. Otherwise, open a new terminal for the `sam_hq_env` environment from Anaconda Navigator.
* In the terminal, execute the following commands to clone the SAM-HQ repository and install it locally:
    ```bash
    git clone [https://github.com/SysCV/sam-hq.git](https://github.com/SysCV/sam-hq.git)
    cd sam-hq
    pip install -e .
    cd ..
    ```
    This will install it as a package named `segment_anything`.

**6. Download SAM_HQ Model Weights**

* You need to download the pre-trained model weights from the SAM-HQ project (e.g., its GitHub repository). These files typically end in `.pth`, such as `sam_hq_vit_h.pth`, `sam_hq_vit_l.pth`, or `sam_hq_vit_b.pth`. Download them to a location on your computer where you can easily find them.

## UI Guide and Parameter Explanations

The main application window is divided into two tabs: "Single Image Analysis" and "Batch Processing".

### 1. Single Image Analysis

This tab is for in-depth analysis of a single image's pixel grayscale distribution.

* **Image Selection**: Click "Select Image" to choose an image.
* **Analyze Button**: Click "Analyze Aggregation Distribution" to start the analysis. A histogram of the aggregation values will be generated on the right.
* **Aggregation Distribution Chart**:
    * **X-axis**: Aggregation Value (0-100), where higher values represent darker pixels.
    * **Y-axis**: Pixel Count.
    * The chart automatically annotates significant Peaks and Valleys.
* **Statistics**: Displays total pixel count, average/min/max aggregation values, and the specific coordinates of peaks/valleys.
* **Save/Clear Chart**: You can save the analysis chart as an image file or clear the current chart.

### 2. Batch Processing

This tab is for automated mask generation and filtering for an entire folder of images.

#### **File Selection**

* **Input Folder**: Click the "Select Input Folder" button to choose a folder containing your source images.
* **Model File**: Click the "Select SAM Model File (.pth)" button to select your downloaded SAM_HQ model weights file.

#### **Configuration**

* **Save Config**: Saves all current filter and SAM parameters to a JSON file.
* **Load Config**: Loads and applies previously saved parameters from a JSON file.

#### **Phase 1: Region Property Filters**

* **Min Area (A) Threshold**: The minimum number of pixels a region must have to be considered. Default: `0`.
* **Min Intensity (I) Threshold**: The sum of aggregation values for all pixels in a region must not be below this value. Default: `0.0`.
* **Min Ratio (R=I/A) Threshold**: The ratio of total intensity (Intensity) to area (Area) must not be below this value. Default: `0.0`.

#### **Phase 2: Pixel Aggregate Filters**

* **Min Aggregate Threshold**: A pixel's aggregation value (range 0-100) must not be below this value. Default: `0`.
* **Max Aggregate Threshold**: A pixel's aggregation value (range 0-100) must not be above this value. Default: `100`.
    * **Filter Logic**: Only pixels with an aggregation value within the set range are used for the final calculation of area, intensity, and ratio. However, if **at least one** such valid pixel exists in a region, the program will draw and keep the **original, complete contour** of that region.

#### **SAM_HQ Auto Mask Generator Parameters**

* **Points Per Side**: The density of points to sample evenly across the image. More points lead to more detailed detection but are slower. Default: `32`.
* **Points Per Batch**: The number of points to process in a single batch. Default: `64`.
* **Pred IoU Thresh**: The threshold for the predicted Intersection over Union (IoU) score, which measures mask quality. Default: `0.88`.
* **Stability Score Thresh**: The threshold for the mask stability score. Default: `0.95`.
* **Stability Score Offset**: An offset parameter for the stability score. Default: `1.0`.
* **Box NMS Thresh**: The Non-Maximum Suppression (NMS) threshold for merging highly overlapping bounding boxes. Default: `0.7`.
* **Crop N Layers**: The number of layers for hierarchical cropping of large images. `0` disables cropping. Default: `0`.
* **Crop NMS Thresh**: The NMS threshold for merging overlapping masks found in different cropped sections. Default: `0.7`.
* **Crop Overlap Ratio**: The overlap ratio between adjacent cropped sections. Default: `0.341` (i.e., 512/1500).
* **Crop Points Downscale Factor**: A downscaling factor for point sampling density on cropped patches. Default: `1`.
* **Min Mask Region Area**: SAM will ignore any regions with an area smaller than this value during generation. Default: `0`.
* **Output Mode**: The output format for the masks. Default: `binary_mask`.
* **Reset SAM_HQ Params to Defaults**: Resets all the SAM parameters above to their built-in default values.

#### **Operations**

* **Start Processing All Images**: Once the input and model are selected, click this to begin batch processing.
* **Open Output Folder**: Opens the directory where results are saved.
* **Export All Values**: After processing is complete, exports all data for the filtered masks to a `.txt` file.

#### **System Monitor & Processing Log**

* These two areas display real-time system resource usage and detailed program execution logs, respectively.

## A Few Tips

* The first time you load a model or process a very large image, it might take a while. Please be patient.
* This tool processes images with a maximum dimension of 4096 pixels on either side. If an image's height or width exceeds this limit, it will be skipped.
* Ensure you have enough disk space to save the processed images and exported data files.
* If the GPU monitor shows "N/A" or an error, make sure your NVIDIA drivers are up to date and the `nvidia-ml-py` (which provides `pynvml`) package is correctly installed in the current Python environment.
* If you have trouble installing `pycocotools` on Windows, try `pip install pycocotools-windows` first.

## How to Contribute

Contributions of all kinds are very welcome!

* **Report Bugs or Suggest Enhancements**: Found a problem or have an idea for an improvement? Please open an issue on the GitHub [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page. The more detail, the better (e.g., steps to reproduce, screenshots, your system and software versions).
* **Contribute Code**: Want to modify the code directly? Great! Please follow the standard GitHub Fork & Pull Request workflow. It's best to open an issue first to discuss the changes you'd like to make.

## Reporting Issues

If you encounter a bug or have a feature request, please open a new issue on the repository's [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page.
