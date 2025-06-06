# SAM_HQ Batch Image Processing and Analysis Tool

If you're looking for a tool that can batch process images, utilize the SAM_HQ (Segment Anything in High Quality) model to generate high-quality image masks, filter these masks according to custom rules, and conveniently save/load processing parameters while monitoring system resources, then this PyQt6-based graphical interface tool might be just what you need.

## What Can It Do?

* **Simple and User-Friendly GUI**: Developed with PyQt6, easy to get started with just a few clicks.
* **Batch Image Processing**: Select a folder, and it will find all images within it (including subfolders) and process them one by one. Supports `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` formats.
* **SAM_HQ Model Support**: You can load your downloaded SAM_HQ pre-trained models (`.pth` files) to automatically generate ultra-high-quality image masks.
* **Adjustable Mask Generation Parameters**: All parameters of the SAM_HQ automatic mask generator can be adjusted according to your needs.
* **Two-Step Mask Filtering**:
    * **Step 1: Region Properties**: Perform initial filtering based on the pixel area (Area), total brightness or aggregate value (Intensity), and the ratio of these two (Ratio = Intensity/Area) for each mask region. These filters can be toggled individually, and thresholds can be customized. Relevant floating-point values are handled with two-decimal precision.
    * **Step 2: Pixel Aggregate Values**: For masks прошедшие the first step, the tool further examines each pixel's aggregate value within the region (calculated as `100 - (grayscale_value / 255.0) * 100.0`), retaining only pixels within a specified range.
* **Results Visualization and Preview**:
    * Outlines of masks that pass the filters, along with their area, total aggregate value, and ratio information (floating-point numbers are kept to two decimal places), are drawn on the original image and saved as a new image.
    * Provides a "View Last Output" feature for quickly checking the most recently processed image.
* **Parameter Configuration Management**:
    * One-click **saving of all current filter parameters and SAM_HQ model parameters** to a JSON file.
    * Easily **load previously saved parameter configuration files** to quickly restore working settings.
* **Clear Data Export**:
    * After processing, structured data for all masks that passed the filters (image name, mask index, area, intensity, ratio) can be exported with one click to a `.txt` file, with numerical values preserved to two decimal places.
* **Organized Output Files**:
    * Processed images are saved in the parent directory of your selected input folder, within a subfolder named according to the current parameters.
    * If images originate from subfolders, the output filename automatically includes relative path information (path separators are replaced with underscores), preventing overwriting of identically named images from different subfolders.
* **Real-time Processing Log and Progress**:
    * The interface displays detailed processing progress, an overall progress bar, and any issues that may arise.
* **System Resource Monitoring**:
    * Real-time display of CPU usage and memory usage on the interface.
    * If the corresponding libraries are installed and an NVIDIA graphics card is present, GPU usage and VRAM usage will also be displayed.
* **Recommended Operating System**: This tool is primarily developed and tested in Windows 10 and Windows 11 environments. Running on these systems is recommended for the best experience.

## Environment Setup

To run this tool, you need a Python environment on your computer and some necessary libraries installed. **Windows 10 or Windows 11 operating system is recommended.**
We will primarily use Anaconda Navigator to create and manage environments, and then use the `pip` command to install packages.

**1. Install Anaconda (if not already on your computer)**

* Go to the [Anaconda official website](https://www.anaconda.com/products/distribution) to download the latest version of Anaconda Distribution for your operating system and complete the installation.

**2. Create an Environment using Anaconda Navigator**

* Open Anaconda Navigator.
* Select the "Environments" tab in the left navigation bar.
* Click the "Create" button at the bottom.
* In the dialog box that appears:
    * **Name**: Enter an environment name, for example, `sam_hq_env`.
    * **Packages**: Ensure Python is selected, and in its version dropdown menu, choose **3.10** or **3.11** (versions compatible with PyTorch are recommended).
* Click "Create" to create the environment. This may take some time.

**3. Install Git (if not already on your computer)**

* Git is used to download code from GitHub. If Git is not yet installed on your computer, please visit [git-scm.com](https://git-scm.com/downloads) to download and install it.

**4. Install Required Python Packages using `pip` (via Terminal)**

* In Anaconda Navigator's "Environments" tab, ensure the `sam_hq_env` environment you just created is selected.
* Click the play button (▶) next to the environment name, then select "Open Terminal". This will open a command-line window with the `sam_hq_env` environment already activated.
* In the opened terminal, use the following `pip install` commands to install the packages:

    * **Install PyTorch (including torchvision and torchaudio):**
        * Visit the [PyTorch official website](https://pytorch.org/get-started/locally/).
        * On the page, select your operating system (Windows), package manager (`Pip`), programming language (Python), and CUDA version (if you have an NVIDIA graphics card and want to use GPU acceleration, select the corresponding CUDA version; e.g., CUDA 11.8 or CUDA 12.1).
        * Copy the `pip` installation command provided by the website and execute it in the terminal. For example, the command for Windows + CUDA 11.8 is typically:
            ```bash
            pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
            ```
            If you don't need GPU or don't have an NVIDIA GPU, select the command for the CPU version.

    * **Install other core dependencies:**
        In the same terminal, continue to execute the following command:
        ```bash
        pip install PyQt6 opencv-python numpy tifffile pycocotools timm
        ```
        * **A small tip about `pycocotools`**: Directly using `pip install pycocotools` on Windows can easily lead to compilation errors. If you encounter this:
            * You can try `pip install pycocotools-windows`.
            * A more reliable method is to go to the [pycocotools page on PyPI](https://pypi.org/project/pycocotools/#files) to find pre-compiled wheel files (`.whl`). Look for a filename that includes your Python version (e.g., `cp311` for Python 3.11) and system architecture (e.g., `win_amd64` for Windows 64-bit). Download it, then use `pip install path_to_your_downloaded_file.whl` to install it.
            * If neither works, you might need to install Microsoft C++ Build Tools to compile from source.

    * **Install optional system monitoring dependencies (recommended):**
        ```bash
        pip install psutil
        pip install nvidia-ml-py # If you have an NVIDIA graphics card and wish to monitor it
        ```

**5. Clone the SAM-HQ Repository and Install Locally (requires Terminal)**

* If the terminal you opened in the previous step is still open and all packages have been installed, you can continue in that terminal. Otherwise, please open a new terminal for the `sam_hq_env` environment from Anaconda Navigator.
* In the opened terminal, execute the following commands to clone the SAM-HQ repository and install it locally:
    ```bash
    git clone [https://github.com/SysCV/sam-hq.git](https://github.com/SysCV/sam-hq.git)
    cd sam-hq
    pip install -e .
    cd ..
    ```
    After this operation, it will be installed as a package named `segment_anything`.

**6. Download SAM_HQ Model Weight Files**

* You need to download pre-trained model weight files from the SAM-HQ project (e.g., its GitHub repository). These files usually end with `.pth`, such as `sam_hq_vit_h.pth`, `sam_hq_vit_l.pth`, or `sam_hq_vit_b.pth`. Download them to a location on your computer where you can easily find them.

## Interface Operations and Parameter Meanings

### 1. File Selection

* **Input Folder**: Click the "Select Input Folder" button to choose a folder containing your original images. The program will automatically find all images in this folder and its subfolders (supports `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` formats).
* **Model File**: Click the "Select SAM Model File (.pth)" button to choose the SAM_HQ model weight file you downloaded earlier.

### 2. Configuration
* **Save Config**: Click this button to save all current "Phase 1 Filter Parameters", "Phase 2 Filter Parameters", and "SAM_HQ Auto Mask Generator Parameters" from the interface to a JSON (`.json`) file. This is convenient for quickly loading settings next time.
* **Load Config**: Click this button to select a previously saved JSON configuration file. The program will automatically apply the parameters from the file to the corresponding controls on the interface.

### 3. Phase 1: Region Property Filters

SAM_HQ generates many initial mask regions; these filters target these regions. Relevant floating-point values (like thresholds for intensity, ratio) are handled with two-decimal precision in the interface and calculations.

* **Min Area (A) Threshold**: A region must have at least this many pixels to be considered. Default: 199.
* **Min Intensity (I) Threshold**: The sum of aggregate values (calculated as `100 - (grayscale_value/255)*100`) of all pixels in a region must not be below this number. Default: 0.00.
* **Min Ratio (R=I/A) Threshold**: The ratio obtained by dividing the total aggregate value (Intensity) by the area (Area) must not be below this number. Default: 44.00.
* **Enable Phase 1 Filters**:
    * **Area (A)**: Check to use area filtering.
    * **Intensity/Ratio (I/R)**: Check to use total aggregate value and ratio filtering.

### 4. Phase 2: Pixel Aggregate Filters

For mask regions that pass the first step of filtering, the tool further examines each pixel within the region.

* **Min Aggregate Threshold**: The aggregate value of a pixel (range 0-100) must not be below this number. Default: 35.
* **Max Aggregate Threshold**: The aggregate value of a pixel (range 0-100) must not be above this number. Default: 100.
    (Only pixels whose aggregate values fall within this set range are considered valid. Final calculations for area, total aggregate value, ratio, and drawing also use only these valid pixels.)

### 5. SAM_HQ Auto Mask Generator Parameters
This section tells the computer how to more intelligently and meticulously find various objects in the image automatically (we call this "generating masks"). You can think of these parameters as adjusting various knobs and buttons on a super camera; proper adjustment results in better "segmentation maps". Relevant floating-point parameters are displayed with two decimal places (some with three, like crop overlap ratio) on the interface.

* **Points Per Side**: Imagine the computer scatters many "observation points" evenly across the image to understand it. This parameter specifies how many points to scatter along each edge of the image. More points mean finer detail but slower processing. Default is 32 points.
* **Points Per Batch**: When processing these "observation points," the computer handles them in batches rather than one by one. This parameter specifies how many points are processed in each batch. Default is 64 points.
* **Pred IoU Thresh (Prediction IoU Threshold)**: After finding an object, the computer scores how accurately it was found. This score is called IoU. This parameter sets a threshold; only objects with scores above this threshold are considered well-found. Default is 0.86 (out of 1).
* **Stability Score Thresh (Stability Score Threshold)**: Sometimes the computer might "hesitate" when finding an object, identifying the same object differently at different times. This parameter also sets a threshold to see how stably the computer finds an object; higher stability is better. Default is 0.92.
* **Stability Score Offset**: This is a small adjustment parameter for the "stability score" mentioned above; generally, you don't need to worry about it much. Default is 1.00.
* **Box NMS Thresh (Box Non-Maximum Suppression Threshold)**: When finding objects, the computer might draw several boxes around the same object. This parameter tells the computer that if several boxes enclose roughly the same thing, remove the redundant ones and keep only the best one. Default is 0.70.
* **Crop N Layers**: If an image is particularly large, the computer might cut it into several smaller pieces and look at them one by one for more detail. This parameter specifies how many layers to cut it into. Default is 1 layer (meaning it's not cut much or only cut once).
* **Crop NMS Thresh (Crop NMS Threshold)**: As mentioned above, when viewing cut pieces, duplicate objects might be found in different small pieces. This parameter is similar to the "Box NMS Threshold" and is used to remove these duplicates. Default is 0.70.
* **Crop Overlap Ratio**: When cutting an image into small pieces, to avoid missing objects at the edges, adjacent small pieces will have a slight overlap. This parameter specifies the proportion of this overlap. Default is 0.341 (SAM-HQ default `512/1500`).
* **Crop Points Downscale Factor**: When scattering "observation points" on the cut small image pieces, the density of the points can be adjusted. This parameter adjusts the density; the larger the number, the sparser the points. Default is 2.
* **Min Mask Region Area**: If an object found by the computer is too small (e.g., only a few pixels large), it might not be useful. This parameter specifies that an object must have at least this much area to be retained. Default is 100 pixels.
* **Output Mode**: After finding objects, the computer records these "found regions" in a certain way. Several different recording formats are available here. Usually, the default `binary_mask` (like a black and white image, where white areas are found and black areas are not) is sufficient.
* **Reset SAM_HQ Params to Defaults**: If you've messed up the parameters above and don't know which settings are good, click this button, and all these "object finding parameters" will revert to their initial settings.

### 6. Operations

* **Start Processing All Images**: After selecting the input folder and model file, click this to start processing all found images. An overall progress bar will be displayed during processing.
* **Stop Processing**: You can click this button at any time during processing to safely stop the current task.
* **Open Output Folder**: Clicking this will directly open the automatically generated output folder. The output folder is usually created in the parent directory of the input folder, and its name will include some parameter information.
* **Export All Values**: After processing is complete, click this button to export the structured data (image name, mask index, area, intensity, ratio) for all masks that passed the filters to a `.txt` file. Intensity and ratio values will be preserved to two decimal places.
* **View Last Output**: Click this button to attempt to open the last successfully processed and saved image using the system's default image viewer for a quick preview.

### 7. System Monitor

* This area will (if relevant libraries are installed) periodically display the following information:
    * **CPU Usage**: Current overall CPU usage percentage.
    * **RAM Usage**: Current memory usage percentage and specific amounts.
    * **GPU Usage**: If an NVIDIA graphics card is present and the `pynvml` library is working correctly, displays the core usage of the first GPU.
    * **VRAM Usage**: If an NVIDIA graphics card is present and the `pynvml` library is working correctly, displays the VRAM usage percentage and specific amounts for the first GPU.
* If the corresponding monitoring libraries (`psutil` or `pynvml`) are not installed or fail to initialize, the respective items will show "N/A" or an error message.

### 8. Processing Log

* This area tells you what the program is doing, the current processing stage, and any warnings or errors that occur. This includes the loading status of system monitoring libraries.

## Some Reminders

* Loading the model for the first time, or processing particularly large images, may take some time. Please be patient.
* Ensure you have enough disk space on your computer to save the processed images and exported data files.
* If you encounter issues installing `pycocotools` on Windows, try finding pre-compiled wheel files first.
* If GPU monitoring shows "N/A" or an error, ensure your NVIDIA drivers are up to date and the `nvidia-ml-py` (which provides `pynvml`) package is correctly installed in the current Python environment.

## How to Contribute

Contributions of all kinds are very welcome!

* **Report Bugs or Suggestions**: Found a problem or have an idea for improvement? Please submit it on the GitHub [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page. The more detailed your description, the better (e.g., how to reproduce, screenshots, your system and software versions, etc.).
* **Contribute Code**: Want to modify the code directly? Welcome! Please follow the standard GitHub Fork & Pull Request process. It's best to open an Issue first to discuss the changes you intend to make.

## Reporting Issues

If you encounter a bug or have a feature request, please open a new Issue directly on the repository's [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page.

