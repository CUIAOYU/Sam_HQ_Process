# SAM_HQ Batch Image Processing and Analysis Tool

If you're looking for a tool that can batch process images, generate high-quality masks using the SAM_HQ (Segment Anything in High Quality) model, filter these masks based on custom rules, and conveniently save/load processing parameters while monitoring system resources, then this PyQt6-based graphical user interface tool might be exactly what you need.

## What can it do?

* **Dual-Function Tabs**: Integrates "Batch Processing" and "Single Image Analysis" modules into one interface.
* **Batch Processing Mode**:
    * **Smart File Traversal**: Select a folder, and it will find all images within it and its subfolders for processing. Supports `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff` formats.
    * **SAM_HQ Model Support**: Load your own pre-trained SAM_HQ models (`.pth` files) to automatically generate ultra-high-quality image masks.
    * **Adjustable Mask Generation Parameters**: All parameters of the SAM_HQ automatic mask generator are tunable to your needs.
    * **Two-Step Mask Filtering**:
        * **Phase 1: Region Properties**: Perform an initial filtering based on the pixel area, total intensity (aggregation value), and the ratio of these two (Intensity/Area) for each mask region.
        * **Phase 2: Pixel Aggregation Value**: For masks that pass Phase 1, the tool further examines the aggregation value (`100 - (grayscale_value / 255.0) * 100.0`) of each pixel for more refined filtering.
    * **Result Visualization & Preview**: The outlines of filtered masks, along with their statistical information, are drawn on the original image and saved. A "View Last Output" feature is available for quick checks.
    * **Clear Data Export**: All structured data (image name, mask index, area, intensity, ratio) of the filtered masks can be exported to a `.txt` file with one click.
* **Single Image Analysis Mode**:
    * **Aggregation Value Distribution Visualization**: Select a single image to analyze and generate a histogram of its "aggregation value" distribution across all pixels.
    * **Key Point Identification**: Automatically identifies and annotates peaks and valleys on the distribution chart to help you understand the image's grayscale distribution characteristics.
    * **Detailed Statistics**: Provides key stats like total pixels, mean/min/max aggregation value, and more.
    * **Chart Export**: The generated analysis chart can be saved as a PNG, PDF, or SVG file.
* **General Features**:
    * **User-Friendly GUI**: Developed with PyQt6 for an easy-to-use, point-and-click experience.
    * **Parameter Configuration Management**: **Save/Load** all filter and SAM_HQ parameters to/from a JSON file with a single click.
    * **Organized Output Files**: Output files are automatically saved in a folder named after the current parameters and can handle same-named files from different subfolders without overwriting.
    * **Real-time Log & Progress**: A detailed processing log, an overall progress bar, and any potential issues are displayed in the interface.
    * **System Resource Monitoring**: Real-time display of CPU, Memory, GPU, and VRAM usage (requires relevant libraries).

## Environment Setup

To run this tool, you need a Python environment with several necessary libraries installed. **Windows 10 or Windows 11 is recommended.** We will primarily use Anaconda Navigator to create and manage the environment, and then use `pip` to install packages.

**1. Install Anaconda (if you haven't already)**

* Go to the [Anaconda official website](https://www.anaconda.com/products/distribution) to download and install the latest version of Anaconda Distribution for your operating system.

**2. Create an Environment using Anaconda Navigator**

* Open Anaconda Navigator.
* Select the "Environments" tab on the left navigation bar.
* Click the "Create" button at the bottom.
* In the dialog box:
    * **Name**: Enter an environment name, e.g., `sam_hq_env`.
    * **Packages**: Ensure Python is selected and choose version **3.10** or **3.11** from its dropdown menu (a version compatible with PyTorch is recommended).
* Click "Create" to build the environment. This may take some time.

**3. Install Git (if you haven't already)**

* Git is used to download code from GitHub. If you don't have Git, please visit [git-scm.com](https://git-scm.com/downloads) to download and install it.

**4. Install Required Python Packages via Terminal**

* In the "Environments" tab of Anaconda Navigator, make sure your newly created `sam_hq_env` is selected.
* Click the play button (▶) next to the environment name and select "Open Terminal". This opens a command line window with the `sam_hq_env` already activated.
* In the terminal, use the following `pip install` commands:

    * **Install PyTorch (including torchvision and torchaudio):**
        * Visit the [PyTorch official website](https://pytorch.org/get-started/locally/).
        * On the page, select your OS (Windows), package manager (`Pip`), language (Python), and CUDA version (if you have an NVIDIA GPU and want to use GPU acceleration, select the corresponding CUDA version, e.g., CUDA 11.8 or 12.1).
        * Copy the provided `pip` install command and execute it in your terminal. For example, for Windows + CUDA 11.8, the command might be:
            ```bash
            pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
            ```
            If you don't need GPU support or don't have an NVIDIA GPU, select the CPU version command.

    * **Install Other Core Dependencies:**
        In the same terminal, continue by running:
        ```bash
        pip install PyQt6 opencv-python numpy tifffile pycocotools timm matplotlib
        ```
        * **A tip for `pycocotools`**: A direct `pip install pycocotools` on Windows can often lead to compilation errors. If this happens, try `pip install pycocotools-windows`.

    * **Install Optional System Monitoring Dependencies (Recommended):**
        ```bash
        pip install psutil
        pip install nvidia-ml-py # If you have an NVIDIA GPU and want to monitor it
        ```

**5. Clone and Install the SAM-HQ Repository**

* If your terminal is still open from the previous step, you can continue. Otherwise, open a new terminal for the `sam_hq_env` from Anaconda Navigator.
* In the terminal, execute the following commands to clone and install the SAM-HQ repository locally:
    ```bash
    git clone [https://github.com/SysCV/sam-hq.git](https://github.com/SysCV/sam-hq.git)
    cd sam-hq
    pip install -e .
    cd ..
    ```
    This will install it as a package named `segment_anything`.

**6. Download SAM_HQ Model Weights**

* You need to download pre-trained model weight files from the SAM-HQ project (e.g., from its GitHub repository). These files usually end with `.pth`, such as `sam_hq_vit_h.pth`, `sam_hq_vit_l.pth`, or `sam_hq_vit_b.pth`. Download them to a convenient location on your computer.

## Interface Operations and Parameter Meanings

The main interface is divided into two tabs: "Single Image Analysis" and "Batch Processing".

### 1. Single Image Analysis

This tab is for in-depth analysis of a single image's pixel grayscale distribution.

* **Image Selection**: Click "Select Image" to choose an image.
* **Analyze Button**: Click "Analyze Aggregation Distribution" to start the analysis. A histogram of the aggregation value distribution will be generated on the right.
* **Aggregation Distribution Chart**:
    * **X-axis**: Aggregation Value (0-100), where a higher value means the pixel is darker.
    * **Y-axis**: Pixel Count.
    * The chart automatically annotates significant peaks and valleys.
* **Statistics**: Displays total pixel count, mean/min/max aggregation values, and the coordinates of peaks/valleys.
* **Save/Clear Chart**: Allows you to save the analysis chart as an image or clear the current chart.

### 2. Batch Processing

This tab is for automated mask generation and filtering for a whole folder of images.

#### **File Selection**

* **Input Folder**: Click the "Select Input Folder" button to choose a folder containing your source images.
* **Model File**: Click the "Select SAM Model File (.pth)" button to choose your downloaded SAM_HQ model weights file.

#### **Configuration**

* **Save Config**: Saves all current filter and SAM parameters to a JSON file.
* **Load Config**: Loads and applies previously saved parameters from a JSON file.

#### **Phase 1: Region Property Filters**

* **Min Area (A) Threshold**: The minimum number of pixels a region must have. Default: `0`.
* **Min Intensity (I) Threshold**: The sum of aggregation values of all pixels in a region must not be below this value. Default: `0.0`.
* **Min Ratio (R=I/A) Threshold**: The ratio of total intensity to area must not be below this value. Default: `0.0`.
* **Enable Phase 1 Filters**:
    * **Area (A)**: Check to enable filtering by area.
    * **Intensity/Ratio (I/R)**: Check to enable filtering by intensity and ratio.

#### **Phase 2: Pixel Aggregate Filters**

* **Min Aggregate Threshold**: A pixel's aggregation value (range 0-100) must not be below this. Default: `0`.
* **Max Aggregate Threshold**: A pixel's aggregation value (range 0-100) must not be above this. Default: `100`.
    * **Filtering Logic**: Only pixels with an aggregation value within this set range are used for calculating the final area, intensity, and ratio. However, as long as a region contains **at least one** such valid pixel, the program will draw the **original, complete outline** of that region and keep it.

#### **SAM_HQ Auto Mask Generator Parameters**

* **Points Per Side**: The density of points sampled on a grid over the image. More points are more detailed but slower. Default: `32`.
* **Points Per Batch**: How many points to process in a single batch. Default: `64`.
* **Pred IoU Thresh**: The predicted IoU threshold for a mask to be considered high quality. Default: `0.88`.
* **Stability Score Thresh**: The stability score threshold for a mask. Default: `0.95`.
* **Stability Score Offset**: An offset for the stability score. Default: `1.0`.
* **Box NMS Thresh**: The threshold for non-maximum suppression of bounding boxes. Default: `0.7`.
* **Crop N Layers**: The number of layers of crops to process for large images. 0 means no cropping. Default: `0`.
* **Crop NMS Thresh**: The NMS threshold for merging masks from different crops. Default: `0.7`.
* **Crop Overlap Ratio**: The overlap ratio between adjacent crops. Default: `0.341` (i.e., 512/1500).
* **Crop Points Downscale Factor**: Downscaling factor for the number of points sampled in crops. Default: `1`.
* **Min Mask Region Area**: Regions smaller than this area will be filtered out during mask generation by SAM. Default: `0`.
* **Output Mode**: The format for the output masks. Default: `binary_mask`.
* **Reset SAM_HQ Params to Defaults**: Click to restore all SAM parameters above to their default values.

#### **Operations**

* **Start Processing All Images**: Once the input folder and model are selected, click this to start processing.
* **Stop Processing**: Safely stops the current task at any time during processing.
* **Open Output Folder**: Directly opens the folder where results are saved.
* **Export All Values**: After processing, exports the structured data of all filtered masks to a `.txt` file.
* **View Last Output**: Opens the last successfully processed image using the system's default viewer.

#### **System Monitor & Processing Log**

* These two areas display real-time system resource usage and a detailed log of the program's operations, respectively.

## A Few Reminders

* Loading the model for the first time or processing very large images may take some time. Please be patient.
* This tool processes images with a maximum single dimension of 4096 pixels. If an image's height or width exceeds this limit, it will be skipped.
* Ensure you have enough disk space to save the processed images and exported data files.
* If the GPU monitor shows "N/A" or an error, ensure your NVIDIA drivers are up to date and the `nvidia-ml-py` (providing `pynvml`) package is correctly installed in the current Python environment.
* If `pycocotools` fails to install on Windows, try `pip install pycocotools-windows` first.

## How to Contribute

Contributions of all forms are very welcome!

* **Submit Bugs or Suggestions**: Found a problem or have an idea for improvement? Please open an issue on the [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page on GitHub. The more detailed your description, the better.
* **Contribute Code**: Want to modify the code directly? Welcome! Please follow the standard GitHub Fork & Pull Request workflow. It's best to open an issue first to discuss the changes you'd like to make.

## Reporting Issues

If you encounter a bug or have a feature request, please open a new issue directly on the repository's [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page.
