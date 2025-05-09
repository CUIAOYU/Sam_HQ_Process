# SAM_HQ Batch Image Processing

If you're looking for a tool to batch process images, generate high-quality image masks using the SAM_HQ (Segment Anything in High Quality) model, and filter these masks according to custom rules, then this PyQt6-based graphical user interface (GUI) tool might be just what you need.

## What Can It Do?

* **Simple and User-Friendly GUI**: Developed with PyQt6, it's easy to get started with just a few clicks.
* **Batch Image Processing**: Select a folder, and it will find all images within it (including subfolders) and process them one by one. Supports `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` formats.
* **SAM_HQ Model Support**: You can load your downloaded SAM_HQ pre-trained models (`.pth` files) to automatically generate ultra-high-quality image masks.
* **Adjustable Mask Generation Parameters**: You can tweak various parameters of the SAM_HQ automatic mask generator, such as `points_per_side`, `pred_iou_thresh`, `stability_score_thresh`, etc., according to your needs.
* **Two-Stage Mask Filtering**:
    * **Stage One: Region Property Filters**: Performs initial filtering based on the pixel area (Area) of each mask region, its total brightness/aggregation value (Intensity), and the ratio of these two (Ratio = Intensity/Area). These filters can be enabled/disabled individually, and their thresholds can be customized.
    * **Stage Two: Pixel Aggregate Filters**: For masks that pass the first stage, the tool further examines each pixel within the region based on its aggregation value (calculated as `100 - (grayscale_value / 255.0) * 100.0`), retaining only pixels within a specified range.
* **Results Visualization**: The contours of the filtered masks, along with their area, total aggregation value, and ratio, are drawn on the original image, and the processed image is saved.
* **Organized Output Files**:
    * Processed images are saved in a subfolder (named according to the current parameters) within the parent directory of your input folder.
    * If images come from subfolders, the output filenames will include their relative path information (with path separators replaced by underscores) to prevent overwriting identically named files from different subfolders.
* **Real-time Processing Log**: The interface displays detailed processing progress and any potential issues or errors.
* **Recommended Operating Systems**: This tool has been primarily developed and tested in Windows 10 and Windows 11 environments. Running it on these systems is recommended for the best experience.

## Environment Setup

To run this tool, you'll need a Python environment and several necessary libraries. **Using Windows 10 or Windows 11 is recommended.**
We'll primarily use Anaconda Navigator to create and manage the environment, and then use `pip` commands to install packages.

**1. Install Anaconda (if you haven't already)**

* Go to the [Anaconda Official Website](https://www.anaconda.com/products/distribution) to download and install the latest Anaconda Distribution for your operating system.

**2. Create an Environment using Anaconda Navigator**

* Open Anaconda Navigator.
* Select the "Environments" tab on the left navigation bar.
* Click the "Create" button at the bottom.
* In the dialog box that appears:
    * **Name**: Enter an environment name, for example, `sam_hq_env`.
    * **Packages**: Ensure Python is selected, and choose version **3.11** from its version dropdown menu.
* Click "Create" to create the environment. This might take some time.

**3. Install Git (if you haven't already)**

* Git is used to download code from GitHub. If you don't have Git installed, please visit [git-scm.com](https://git-scm.com/downloads) to download and install it.

**4. Install Required Python Packages using `pip` (via Terminal)**

* In Anaconda Navigator, under the "Environments" tab, make sure the `sam_hq_env` environment you just created is selected.
* Click the play button (▶) next to the environment name, then select "Open Terminal". This will open a command-line window with the `sam_hq_env` environment already activated.
* In the opened terminal, use the following `pip install` commands to install the packages:

    * **Install PyTorch (including torchvision and torchaudio):**
        * Visit the [PyTorch Official Website](https://pytorch.org/get-started/locally/).
        * On the page, select your operating system (Windows), package manager (`Pip`), programming language (Python), and CUDA version (if you have an NVIDIA GPU and want GPU acceleration, choose the corresponding CUDA version; CUDA 12.1 is a good choice).
        * Copy the `pip` installation command provided by the website and execute it in the terminal. For example, the command for Windows + CUDA 12.1 is typically:
            ```bash
            pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
            ```
            If you don't need GPU support or don't have an NVIDIA GPU, select the CPU version command.

    * **Install other dependencies:**
        In the same terminal, continue by executing the following command:
        ```bash
        pip install PyQt6 opencv-python numpy tifffile pycocotools timm
        ```
        * **A note on `pycocotools`**: Directly using `pip install pycocotools` on Windows can sometimes lead to compilation errors. If you encounter this:
            * You can try `pip install pycocotools-windows` (a community-maintained package, though its availability might vary).
            * A more reliable method is to visit the [pycocotools page on PyPI](https://pypi.org/project/pycocotools/#files) and look for a precompiled wheel file (`.whl`) that matches your Python version (e.g., `cp311` for Python 3.11) and system architecture (e.g., `win_amd64` for Windows 64-bit). Download it, and then install it using `pip install path/to/your/downloaded_file.whl`.
            * If all else fails, you might need to install Microsoft C++ Build Tools to compile it from source.

**5. Clone the SAM-HQ Repository and Install it Locally (requires Terminal)**

* If the terminal from the previous step is still open and all packages are installed, you can continue there. Otherwise, open a new terminal for the `sam_hq_env` environment via Anaconda Navigator.
* In the opened terminal, execute the following commands to clone the SAM-HQ repository and install it locally:
    ```bash
    git clone [https://github.com/SysCV/sam-hq.git](https://github.com/SysCV/sam-hq.git)
    cd sam-hq
    pip install -e .
    cd ..
    ```
    This will install it as a package named `segment_anything`.

**6. Download SAM_HQ Model Weights**

* You need to download pre-trained model weight files from the SAM-HQ project (e.g., its GitHub repository). These files usually have a `.pth` extension, such as `sam_hq_vit_h.pth`, `sam_hq_vit_l.pth`, or `sam_hq_vit_b.pth`. Save them to a location on your computer where you can easily find them.

## User Interface and Parameter Explanations

### 1. File Selection

* **Input Folder**: Click the "Select Input Folder" button to choose the folder containing your source images. The program will automatically search for images (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`) in this folder and all its subfolders.
* **Model File**: Click the "Select SAM Model File (.pth)" button to choose the SAM_HQ model weight file you downloaded.

### 2. Phase 1: Region Property Filters

These filters are applied to each initial mask region generated by SAM_HQ.

* **Min Area (A) Threshold**: The minimum pixel area a region must have to be considered. Default: 199.
* **Min Intensity (I) Threshold**: The minimum sum of aggregated pixel values (calculated as `100 - (grayscale_value/255)*100`) within a region. Default: 0.0.
* **Min Ratio (R=I/A) Threshold**: The minimum ratio of the above Intensity to Area. Default: 44.0.
* **Enable Phase 1 Filters**:
    * **Area (A)**: Check to enable area filtering.
    * **Intensity/Ratio (I/R)**: Check to enable total aggregation value and ratio filtering.

### 3. Phase 2: Pixel Aggregate Filters

These filters are applied to each pixel within the mask regions that passed Phase 1.

* **Min Aggregate Threshold**: The minimum aggregation value (range 0-100) a pixel must have. Default: 35.
* **Max Aggregate Threshold**: The maximum aggregation value (range 0-100) a pixel can have. Default: 100.
    (Only pixels with aggregation values within this range are considered valid and used for final area/intensity/ratio calculations and visualization.)

### 4. SAM_HQ Auto Mask Generator Parameters
This section lets you tell the computer how to more intelligently and meticulously find various objects in your images (this process is called "generating masks"). You can think of these parameters as various knobs and buttons on a super-camera; adjusting them correctly can help you get better "segmentation maps."

* **Points Per Side**: Imagine the computer scatters many "observation points" evenly across the image to understand it. This parameter sets how many points are placed along each edge of the image. More points mean finer detail but slower processing. Default is 32 points.
* **Points Per Batch**: When processing these "observation points," the computer handles them in batches rather than one by one. This parameter sets how many points are in each batch. Default is 64 points.
* **Pred IoU Thresh**: After finding an object, the computer scores how accurately it was found. This score is called IoU. This parameter sets a threshold; only objects with a score above this threshold are considered well-found. Default is 0.86 (out of 1).
* **Stability Score Thresh**: Sometimes the computer might "hesitate" when finding an object, seeing it differently at different times. This parameter sets a threshold for how stably the computer identifies an object; higher stability is better. Default is 0.92.
* **Stability Score Offset**: This is a small adjustment parameter for the "stability score" mentioned above; you usually don't need to worry about it. Default is 1.0.
* **Box NMS Thresh**: When finding objects, the computer might draw several bounding boxes around the same object. This parameter tells the computer to remove redundant boxes if they largely overlap, keeping only the best one. Default is 0.7.
* **Crop N Layers**: If an image is very large, the computer might cut it into smaller pieces to examine each piece more closely. This parameter sets how many "layers" of cuts to make. Default is 1 layer (meaning minimal or no cutting).
* **Crop NMS Thresh**: Similar to "Box NMS Thresh," this parameter is used to remove duplicate findings when objects are found in different cropped sections. Default is 0.7.
* **Crop Overlap Ratio**: When cutting the image into smaller pieces, adjacent pieces will overlap slightly to avoid missing objects at the edges. This parameter sets the proportion of this overlap. Default is 0.341 (about one-third).
* **Crop Points Downscale Factor**: When scattering "observation points" on the smaller, cropped image pieces, you can adjust the density of these points. This parameter adjusts the density; a larger number means sparser points. Default is 2.
* **Min Mask Region Area**: If an object found by the computer is too small (e.g., only a few pixels), it might not be useful. This parameter sets the minimum area an object must have to be kept. Default is 100 pixels.
* **Output Mode**: After finding objects, the computer records these "found regions" in a specific format. There are several formats to choose from. The default `binary_mask` (like a black and white image, where white areas are found objects and black areas are not) is usually fine.
* **Reset SAM_HQ Params to Defaults**: If you've messed up the parameters above and aren't sure what's best, click this button, and all these "object finding parameters" will revert to their initial settings.

### 5. Operations

* **Start Processing All Images**: After selecting the input folder and model file, click this button to start processing all found images.
* **Stop Processing**: You can click this button at any time during processing to safely stop the current task.
* **Open Output Folder**: Click this to directly open the auto-generated output folder. The output folder is usually created in the parent directory of the input folder and named based on the parameters used.

### 6. Processing Log

* This area will show you detailed information about what the program is doing, its progress, and any warnings or errors.

## Important Notes

* Loading the model for the first time or processing very large images might take a while. Please be patient.
* Ensure you have enough disk space to save the processed images.
* If you encounter issues installing `pycocotools` on Windows, try using a precompiled wheel file first.
* Correctly setting the `PYTHONPATH` is crucial for Python to find the `segment_anything` (SAM-HQ) library installed from your local repository.

## How to Contribute

Contributions of all kinds are very welcome!

* **Report Bugs or Suggest Enhancements**: Found a problem or have an idea for improvement? Please open an issue on the GitHub [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page. The more detail, the better (e.g., how to reproduce, screenshots, your system and software versions).
* **Contribute Code**: Want to modify the code directly? Welcome! Please follow the standard GitHub Fork & Pull Request workflow. It's best to open an Issue first to discuss the changes you'd like to make.

## Reporting Issues

If you encounter a bug or have a feature request, please open a new Issue directly on the repository's [Issues](https://github.com/CUIAOYU/Useful-Image-Analysis-Tool-Integration/issues) page.

