import sys
import os
import cv2
import numpy as np
import torch
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="timm\.models\.layers")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm\.models\.registry")
warnings.filterwarnings("ignore", category=UserWarning, module="segment_anything\.modeling\.tiny_vit_sam")

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QGroupBox, QTextEdit, QSpinBox, QCheckBox,
                             QDoubleSpinBox, QFormLayout, QScrollArea, QMessageBox, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt6.QtGui import QPixmap, QImage

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SEGMENT_ANYTHING_AVAILABLE = True
except ImportError as e:
    print("Error: 'segment_anything' library not found.")
    print("Please ensure it and its dependencies (like 'timm') are correctly installed according to the environment setup guide.")
    SEGMENT_ANYTHING_AVAILABLE = False

try:
    from pycocotools import mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: 'pycocotools' library not found. RLE output modes ('uncompressed_rle', 'coco_rle') decoding will be unavailable.")

try:
    import tifffile
except ImportError:
    tifffile = None
    print("Warning: 'tifffile' library not found. TIFF images might not load correctly.")


class HQSAMProcessorThread(QThread):
    processing_finished = pyqtSignal()
    log_updated = pyqtSignal(str)

    MAX_IMAGE_SIZE = 4096

    bright_colors = [
        (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
        (255, 0, 0), (147, 20, 255), (255, 0, 255), (255, 255, 0),
        (128, 0, 128),
    ]

    def __init__(self, input_folder, output_folder, model_path, filter_params, sam_params):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_path = model_path
        self.filter_params = filter_params
        self.sam_params = sam_params
        self.running = True
        self.lock = QMutex()

    def validate_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.access(image_path, os.R_OK):
            raise PermissionError(f"No read permission for: {image_path}")

    def run(self):
        if not SEGMENT_ANYTHING_AVAILABLE:
            self.log_updated.emit("Error: 'segment_anything' library is unavailable, cannot proceed.")
            self.processing_finished.emit()
            return

        try:
            if not os.path.exists(self.input_folder):
                raise FileNotFoundError(f"Input folder not found: {self.input_folder}")

            os.makedirs(self.output_folder, exist_ok=True)
            if not os.access(self.output_folder, os.W_OK):
                raise PermissionError(f"No write permission for: {self.output_folder}")

            self.log_updated.emit(f"Output folder: {self.output_folder}")
            self.log_updated.emit("Loading SAM_HQ model...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_updated.emit(f"Using device: {device}")

            try:
                if not os.path.exists(self.model_path): raise FileNotFoundError(f"SAM model file not found: {self.model_path}")
                if not os.access(self.model_path, os.R_OK): raise PermissionError(f"No read permission for SAM model file: {self.model_path}")
                model_type = "vit_h"; model_filename_lower = os.path.basename(self.model_path).lower()
                if "vit_l" in model_filename_lower: model_type = "vit_l"
                elif "vit_b" in model_filename_lower: model_type = "vit_b"
                self.log_updated.emit(f"Attempting to load SAM model type: {model_type} (HQ version)")
                sam = sam_model_registry[model_type](checkpoint=self.model_path).to(device); sam.eval()
            except Exception as e:
                raise RuntimeError(f"Failed to load SAM model: {str(e)}")

            mask_generator = SamAutomaticMaskGenerator(model=sam, **self.sam_params)

            self.log_updated.emit("SAM_HQ Generator Parameters:")
            for key, value in self.sam_params.items(): self.log_updated.emit(f"  - {key}: {value}")
            self.log_updated.emit("Filter Parameters:")
            for key, value in self.filter_params.items(): self.log_updated.emit(f"  - {key}: {value}")

            image_files_with_paths = []
            for dirpath, dirnames, filenames in os.walk(self.input_folder):
                for filename in filenames:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                        image_files_with_paths.append(os.path.join(dirpath, filename))
            
            files = image_files_with_paths
            if not files: 
                self.log_updated.emit("Warning: No image files found in the input folder or its subfolders.")
                self.processing_finished.emit()
                return

            self.log_updated.emit(f"Found {len(files)} image file(s) in total (including subfolders).")
            processed_count = 0

            for idx, full_input_path in enumerate(files):
                self.lock.lock()
                is_running = self.running
                self.lock.unlock()
                if not is_running: self.log_updated.emit("Processing stopped by user."); break

                try:
                    input_path = full_input_path
                    self.validate_image(input_path)
                    
                    relative_image_path = os.path.relpath(input_path, self.input_folder)
                    output_base_name_parts = []
                    
                    if os.path.dirname(relative_image_path) != "":
                        output_base_name_parts.append(os.path.dirname(relative_image_path).replace(os.sep, "_"))
                    
                    output_base_name_parts.append(os.path.splitext(os.path.basename(relative_image_path))[0])
                    output_base_name = "_".join(output_base_name_parts)
                    
                    file_ext = os.path.splitext(input_path)[1]
                    
                    output_filename = f"{output_base_name}_sam_hq_processed{file_ext}"
                    output_path = os.path.join(self.output_folder, output_filename)
                    
                    self.log_updated.emit(f"Processing image {idx + 1}/{len(files)}: {input_path}")
                    start_time = time.time()
                    success = self.process_image_with_filter(input_path, output_path, mask_generator)
                    elapsed_time = time.time() - start_time
                    if success:
                        processed_count += 1
                        self.log_updated.emit(f"Processed image saved to: {output_path}")
                    self.log_updated.emit(f"Processing time: {elapsed_time:.2f} seconds")
                except Exception as e:
                    self.log_updated.emit(f"Error processing {os.path.basename(input_path)}: {str(e)}")
                    continue

            self.lock.lock()
            is_running_final = self.running
            self.lock.unlock()
            if is_running_final: self.log_updated.emit(f"Completed: {processed_count}/{len(files)} image(s) processed.")

        except Exception as e:
            self.log_updated.emit(f"Fatal error during processing: {str(e)}")
        finally:
            if 'sam' in locals() and device == 'cuda': del sam; torch.cuda.empty_cache(); self.log_updated.emit("CUDA cache cleared.")
            self.processing_finished.emit()

    def process_image_with_filter(self, input_path, output_path, mask_generator):
        try:
            original_image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
            grayscale_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            if original_image_bgr is None or grayscale_image is None:
                 if tifffile:
                     try:
                         tiff_image = tifffile.imread(input_path)
                         if len(tiff_image.shape) == 3 and tiff_image.shape[2] >= 3: original_image_bgr = cv2.cvtColor(tiff_image[:,:,:3], cv2.COLOR_RGB2BGR); grayscale_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2GRAY)
                         elif len(tiff_image.shape) == 2: grayscale_image = tiff_image; original_image_bgr = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
                         else: raise ValueError("Unsupported TIFF structure")
                         if original_image_bgr.dtype != np.uint8: original_image_bgr = ((original_image_bgr / np.max(original_image_bgr)) * 255).astype(np.uint8) if np.max(original_image_bgr) > 0 else original_image_bgr.astype(np.uint8)
                         if grayscale_image.dtype != np.uint8: grayscale_image = ((grayscale_image / np.max(grayscale_image)) * 255).astype(np.uint8) if np.max(grayscale_image) > 0 else grayscale_image.astype(np.uint8)
                     except Exception as tiff_e: self.log_updated.emit(f"Warning: Failed to read TIFF: {tiff_e}"); return False
                 else: self.log_updated.emit(f"Error: Failed to read image (cv2 failed, tifffile not installed)"); return False
            if original_image_bgr is None or grayscale_image is None: self.log_updated.emit(f"Error: Could not load image"); return False
            if original_image_bgr.shape[0] > self.MAX_IMAGE_SIZE or original_image_bgr.shape[1] > self.MAX_IMAGE_SIZE: self.log_updated.emit(f"Error: Image dimensions too large"); return False

            self.log_updated.emit(f"  Generating masks for {os.path.basename(input_path)} (HQ)...")
            try:
                masks = mask_generator.generate(original_image_bgr)
                self.log_updated.emit(f"  Found {len(masks)} initial masks.")
            except Exception as e:
                self.log_updated.emit(f"Error generating masks: {str(e)}"); return False

            output_image = original_image_bgr.copy()
            height, width = output_image.shape[:2]
            drawn_text_boxes = []
            filtered_mask_count = 0
            use_area_filter = self.filter_params.get('use_area', True); use_intensity_filter = self.filter_params.get('use_intensity', True)
            a_threshold = self.filter_params.get('a_threshold', 0); i_threshold = self.filter_params.get('i_threshold', 0.0); r_threshold = self.filter_params.get('r_threshold', 0.0)
            min_aggregate = self.filter_params.get('min_aggregate', 0); max_aggregate = self.filter_params.get('max_aggregate', 100)

            for i, mask_info in enumerate(masks):
                segmentation_data = mask_info["segmentation"]
                segmentation_mask_bool = None; segmentation_mask_uint8 = None
                if isinstance(segmentation_data, dict):
                    if PYCOCOTOOLS_AVAILABLE:
                        try: segmentation_mask_uint8 = mask_utils.decode(segmentation_data); segmentation_mask_bool = segmentation_mask_uint8.astype(bool)
                        except Exception as decode_e: self.log_updated.emit(f"  Warning: Failed to decode RLE mask ({decode_e}), skipping."); continue
                    else: self.log_updated.emit(f"  Error: Detected RLE mask but 'pycocotools' is not installed."); continue
                elif isinstance(segmentation_data, np.ndarray):
                    segmentation_mask_bool = segmentation_data.astype(bool); segmentation_mask_uint8 = segmentation_mask_bool.astype(np.uint8) * 255
                else: self.log_updated.emit(f"  Warning: Unknown mask format type ({type(segmentation_data)}), skipping."); continue
                if segmentation_mask_bool is None or segmentation_mask_uint8 is None: continue

                region_pixels_gray = grayscale_image[segmentation_mask_bool];
                if len(region_pixels_gray) == 0: continue
                aggregate_values_0_100 = 100.0 - (region_pixels_gray.astype(np.float32) / 255.0) * 100.0
                initial_area = len(aggregate_values_0_100); initial_intensity = np.sum(aggregate_values_0_100); initial_ratio = initial_intensity / initial_area if initial_area > 0 else 0.0
                passed_phase1 = True
                if use_area_filter and initial_area < a_threshold: passed_phase1 = False
                if passed_phase1 and use_intensity_filter and initial_intensity < i_threshold: passed_phase1 = False
                if passed_phase1 and use_area_filter and use_intensity_filter and initial_ratio < r_threshold: passed_phase1 = False
                if not passed_phase1: continue
                pixels_within_agg_range_mask = ((aggregate_values_0_100 >= min_aggregate) & (aggregate_values_0_100 <= max_aggregate))
                num_pixels_passed_phase2 = np.count_nonzero(pixels_within_agg_range_mask)

                if num_pixels_passed_phase2 > 0:
                    filtered_mask_count += 1
                    final_aggregates = aggregate_values_0_100[pixels_within_agg_range_mask]; final_area = num_pixels_passed_phase2
                    final_intensity = np.sum(final_aggregates); final_ratio = final_intensity / final_area if final_area > 0 else 0.0
                    value_text_parts = []
                    if use_area_filter: value_text_parts.append(f"A:{final_area}")
                    if use_intensity_filter:
                        value_text_parts.append(f"I:{final_intensity:.0f}")
                        if use_area_filter and final_area > 0: value_text_parts.append(f"R:{final_ratio:.2f}")
                        elif use_area_filter: value_text_parts.append("R:N/A")
                    value_text = " ".join(value_text_parts) if value_text_parts else "Passed"

                    contours, _ = cv2.findContours(segmentation_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    color_bgr = self.bright_colors[filtered_mask_count % len(self.bright_colors)]
                    cv2.drawContours(output_image, contours, -1, color_bgr, 1)
                    ys, xs = np.where(segmentation_mask_bool);
                    if len(xs) == 0 or len(ys) == 0: continue
                    center_x = int(np.mean(xs)); center_y = int(np.mean(ys))
                    text_x = max(5, min(center_x, width - 10)); text_y = max(15, min(center_y, height - 5))
                    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.4; thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(value_text, font, font_scale, thickness)
                    current_text_box = [text_x, text_y - text_height, text_width, text_height + baseline]; shift_amount = text_height + 2
                    attempts = 0; max_attempts = 10
                    while any(self.boxes_overlap(current_text_box, box) for box in drawn_text_boxes) and attempts < max_attempts:
                        text_y += shift_amount
                        if text_y > height - 5: text_y = max(15, min(center_y, height - 5)); text_y -= (attempts + 1) * shift_amount;
                        if text_y < 15: text_y = max(15, min(center_y, height - 5)); break
                        current_text_box = [text_x, text_y - text_height, text_width, text_height + baseline]; attempts += 1
                    cv2.putText(output_image, value_text, (text_x, text_y), font, font_scale, color_bgr, thickness, cv2.LINE_AA)
                    drawn_text_boxes.append(current_text_box)

            self.log_updated.emit(f"  Kept {filtered_mask_count} masks for visualization after filtering.")
            try:
                cv2.imwrite(output_path, output_image)
                return True
            except Exception as e:
                self.log_updated.emit(f"Error: Failed to save output image {output_path}: {str(e)}")
                return False

        except Exception as e:
            self.log_updated.emit(f"General error processing {os.path.basename(input_path)}: {str(e)}")
            if 'cuda' in str(e).lower() and torch.cuda.is_available(): torch.cuda.empty_cache()
            return False

    def boxes_overlap(self, box1, box2):
        x1, y1, w1, h1 = box1; x2, y2, w2, h2 = box2
        if (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1): return False
        return True

    def stop(self):
        self.lock.lock()
        try:
            self.running = False
            self.log_updated.emit("Attempting to stop processing...")
        finally:
            self.lock.unlock()


class HQSAMMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_folder = None
        self.model_path = None
        self.process_thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('SAM_HQ Image Processing Tool')
        self.setGeometry(100, 100, 1100, 750)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)
        control_scroll_area = QScrollArea(); control_scroll_area.setWidgetResizable(True)
        control_scroll_widget = QWidget(); control_scroll_layout = QVBoxLayout(control_scroll_widget)
        control_scroll_layout.setContentsMargins(5,5,5,5); control_scroll_layout.setSpacing(15)

        file_group = QGroupBox("File Selection"); file_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        file_layout = QVBoxLayout(file_group); file_layout.setContentsMargins(10, 15, 10, 10); file_layout.setSpacing(10)
        input_folder_layout = QHBoxLayout(); self.input_folder_label = QLabel("Input Folder: Not selected")
        input_folder_btn = QPushButton("Select Input Folder"); input_folder_btn.setStyleSheet(self.get_button_style("green")); input_folder_btn.clicked.connect(self.select_input_folder)
        input_folder_layout.addWidget(self.input_folder_label, 1); input_folder_layout.addWidget(input_folder_btn); file_layout.addLayout(input_folder_layout)
        model_file_layout = QHBoxLayout(); self.model_path_label = QLabel("Model File: Not selected")
        model_path_btn = QPushButton("Select SAM Model File (.pth)"); model_path_btn.setStyleSheet(self.get_button_style("green")); model_path_btn.clicked.connect(self.select_model_file)
        model_file_layout.addWidget(self.model_path_label, 1); model_file_layout.addWidget(model_path_btn); file_layout.addLayout(model_file_layout)
        control_scroll_layout.addWidget(file_group)

        param_group_phase1 = QGroupBox("Phase 1: Region Property Filters"); param_group_phase1.setStyleSheet("QGroupBox { font-weight: bold; }")
        param_layout_phase1 = QFormLayout(param_group_phase1); param_layout_phase1.setContentsMargins(10, 15, 10, 10); param_layout_phase1.setVerticalSpacing(10); param_layout_phase1.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.a_threshold_spin = QSpinBox(); self.a_threshold_spin.setRange(0, 1000000); self.a_threshold_spin.setValue(199)
        param_layout_phase1.addRow("Min Area (A) Threshold:", self.a_threshold_spin)
        self.i_threshold_spin = QDoubleSpinBox(); self.i_threshold_spin.setRange(0.0, 100000000.0); self.i_threshold_spin.setValue(0.0); self.i_threshold_spin.setDecimals(0)
        param_layout_phase1.addRow("Min Intensity (I) Threshold:", self.i_threshold_spin)
        self.r_threshold_spin = QDoubleSpinBox(); self.r_threshold_spin.setRange(0.0, 100.0); self.r_threshold_spin.setSingleStep(0.1); self.r_threshold_spin.setValue(44.0); self.r_threshold_spin.setDecimals(1)
        param_layout_phase1.addRow("Min Ratio (R=I/A) Threshold:", self.r_threshold_spin)
        metrics_layout = QHBoxLayout(); metrics_layout.setSpacing(15)
        self.area_check = QCheckBox("Area (A)"); self.area_check.setStyleSheet("font-weight: normal;"); self.area_check.setChecked(True); metrics_layout.addWidget(self.area_check)
        self.intensity_check = QCheckBox("Intensity/Ratio (I/R)"); self.intensity_check.setStyleSheet("font-weight: normal;"); self.intensity_check.setChecked(True); metrics_layout.addWidget(self.intensity_check)
        metrics_layout.addStretch(); param_layout_phase1.addRow("Enable Phase 1 Filters:", metrics_layout); control_scroll_layout.addWidget(param_group_phase1)

        param_group_phase2 = QGroupBox("Phase 2: Pixel Aggregate Filters"); param_group_phase2.setStyleSheet("QGroupBox { font-weight: bold; }")
        param_layout_phase2 = QFormLayout(param_group_phase2); param_layout_phase2.setContentsMargins(10, 15, 10, 10); param_layout_phase2.setVerticalSpacing(10); param_layout_phase2.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.min_aggregate_spin = QSpinBox(); self.min_aggregate_spin.setRange(0, 100); self.min_aggregate_spin.setValue(35)
        param_layout_phase2.addRow("Min Aggregate Threshold:", self.min_aggregate_spin)
        self.max_aggregate_spin = QSpinBox(); self.max_aggregate_spin.setRange(0, 100); self.max_aggregate_spin.setValue(100)
        param_layout_phase2.addRow("Max Aggregate Threshold:", self.max_aggregate_spin); control_scroll_layout.addWidget(param_group_phase2)

        sam_params_group = QGroupBox("SAM_HQ Auto Mask Generator Parameters"); sam_params_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        sam_params_layout = QFormLayout(sam_params_group); sam_params_layout.setContentsMargins(10, 15, 10, 10); sam_params_layout.setVerticalSpacing(10); sam_params_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.points_per_side_spin = QSpinBox(); self.points_per_side_spin.setRange(1, 100); self.points_per_side_spin.setValue(32); sam_params_layout.addRow("Points Per Side:", self.points_per_side_spin)
        self.points_per_batch_spin = QSpinBox(); self.points_per_batch_spin.setRange(1, 1000); self.points_per_batch_spin.setValue(64); sam_params_layout.addRow("Points Per Batch:", self.points_per_batch_spin)
        self.pred_iou_thresh_spin = QDoubleSpinBox(); self.pred_iou_thresh_spin.setRange(0.0, 1.0); self.pred_iou_thresh_spin.setSingleStep(0.01); self.pred_iou_thresh_spin.setValue(0.86); sam_params_layout.addRow("Pred IoU Thresh:", self.pred_iou_thresh_spin)
        self.stability_score_thresh_spin = QDoubleSpinBox(); self.stability_score_thresh_spin.setRange(0.0, 1.0); self.stability_score_thresh_spin.setSingleStep(0.01); self.stability_score_thresh_spin.setValue(0.92); sam_params_layout.addRow("Stability Score Thresh:", self.stability_score_thresh_spin)
        self.stability_score_offset_spin = QDoubleSpinBox(); self.stability_score_offset_spin.setRange(0.0, 10.0); self.stability_score_offset_spin.setSingleStep(0.1); self.stability_score_offset_spin.setValue(1.0); sam_params_layout.addRow("Stability Score Offset:", self.stability_score_offset_spin)
        self.box_nms_thresh_spin = QDoubleSpinBox(); self.box_nms_thresh_spin.setRange(0.0, 1.0); self.box_nms_thresh_spin.setSingleStep(0.01); self.box_nms_thresh_spin.setValue(0.7); sam_params_layout.addRow("Box NMS Thresh:", self.box_nms_thresh_spin)
        self.crop_n_layers_spin = QSpinBox(); self.crop_n_layers_spin.setRange(0, 5); self.crop_n_layers_spin.setValue(1); sam_params_layout.addRow("Crop N Layers:", self.crop_n_layers_spin)
        self.crop_nms_thresh_spin = QDoubleSpinBox(); self.crop_nms_thresh_spin.setRange(0.0, 1.0); self.crop_nms_thresh_spin.setSingleStep(0.01); self.crop_nms_thresh_spin.setValue(0.7); sam_params_layout.addRow("Crop NMS Thresh:", self.crop_nms_thresh_spin)
        self.crop_overlap_ratio_spin = QDoubleSpinBox(); self.crop_overlap_ratio_spin.setRange(0.0, 1.0); self.crop_overlap_ratio_spin.setSingleStep(0.01); self.crop_overlap_ratio_spin.setValue(round(512 / 1500, 3)); sam_params_layout.addRow("Crop Overlap Ratio:", self.crop_overlap_ratio_spin)
        self.crop_n_points_downscale_factor_spin = QSpinBox(); self.crop_n_points_downscale_factor_spin.setRange(1, 10); self.crop_n_points_downscale_factor_spin.setValue(2); sam_params_layout.addRow("Crop Points Downscale Factor:", self.crop_n_points_downscale_factor_spin)
        self.min_mask_region_area_spin = QSpinBox(); self.min_mask_region_area_spin.setRange(0, 100000); self.min_mask_region_area_spin.setValue(100); sam_params_layout.addRow("Min Mask Region Area:", self.min_mask_region_area_spin)
        self.output_mode_combo = QComboBox(); self.output_mode_combo.addItems(["binary_mask", "uncompressed_rle", "coco_rle"]); sam_params_layout.addRow("Output Mode:", self.output_mode_combo)
        reset_sam_params_btn = QPushButton("Reset SAM_HQ Params to Defaults"); reset_sam_params_btn.setStyleSheet(self.get_button_style("grey")); reset_sam_params_btn.clicked.connect(self.reset_sam_params); sam_params_layout.addRow("", reset_sam_params_btn)
        control_scroll_layout.addWidget(sam_params_group)

        control_scroll_layout.addStretch(1); control_scroll_area.setWidget(control_scroll_widget); left_layout.addWidget(control_scroll_area)

        operation_layout = QHBoxLayout()
        self.process_btn = QPushButton("Start Processing All Images"); self.process_btn.setStyleSheet(self.get_button_style("blue")); self.process_btn.setEnabled(False); self.process_btn.clicked.connect(self.start_processing); operation_layout.addWidget(self.process_btn)
        self.stop_btn = QPushButton("Stop Processing"); self.stop_btn.setStyleSheet(self.get_button_style("red")); self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop_processing); operation_layout.addWidget(self.stop_btn)
        self.open_output_btn = QPushButton("Open Output Folder"); self.open_output_btn.setStyleSheet(self.get_button_style("grey")); self.open_output_btn.setEnabled(False); self.open_output_btn.clicked.connect(self.open_output_folder); operation_layout.addWidget(self.open_output_btn)
        left_layout.addLayout(operation_layout)
        main_layout.addWidget(left_widget, 1)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)
        log_group = QGroupBox("Processing Log"); log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout = QVBoxLayout(log_group); log_layout.setContentsMargins(10, 15, 10, 10)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setStyleSheet(self.get_log_style())
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group)
        main_layout.addWidget(right_widget, 1)

        self.update_button_states()

    def get_button_style(self, color="blue"):
        base_style = "QPushButton { padding: 8px 12px; color: white; border: none; border-radius: 4px; font-weight: bold; } QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        colors = {"green": ("#4CAF50", "#45a049"), "blue": ("#2196F3", "#0b7dda"), "red": ("#f44336", "#d32f2f"), "grey": ("#607d8b", "#455a64")}
        bg_color, hover_color = colors.get(color, colors["blue"])
        return base_style + f"QPushButton {{ background-color: {bg_color}; }} QPushButton:hover:!disabled {{ background-color: {hover_color}; }}"

    def get_log_style(self):
        return "QTextEdit { border: 1px solid #cccccc; border-radius: 4px; padding: 5px; font-family: Consolas, Courier New, monospace; background-color: #f8f8f8; }"

    def get_processing_output_folder_path(self):
        if not self.input_folder: return None
        input_folder_name = os.path.basename(os.path.normpath(self.input_folder))
        parent_dir = os.path.dirname(os.path.normpath(self.input_folder))
        param_parts = []
        if self.area_check.isChecked(): param_parts.append(f"A={self.a_threshold_spin.value()}")
        if self.intensity_check.isChecked():
            param_parts.append(f"I={self.i_threshold_spin.value():.0f}")
            param_parts.append(f"R={self.r_threshold_spin.value():.1f}")
        param_parts.append(f"Agg={self.min_aggregate_spin.value()}-{self.max_aggregate_spin.value()}")
        param_str = " ".join(param_parts)
        output_folder_name = f"{input_folder_name}_SAM_HQ_Output [{param_str}]"
        return os.path.join(parent_dir, output_folder_name)

    def reset_sam_params(self):
        self.points_per_side_spin.setValue(32); self.points_per_batch_spin.setValue(64); self.pred_iou_thresh_spin.setValue(0.86); self.stability_score_thresh_spin.setValue(0.92); self.stability_score_offset_spin.setValue(1.0); self.box_nms_thresh_spin.setValue(0.7); self.crop_n_layers_spin.setValue(1); self.crop_nms_thresh_spin.setValue(0.7); self.crop_overlap_ratio_spin.setValue(round(512 / 1500, 3)); self.crop_n_points_downscale_factor_spin.setValue(2); self.min_mask_region_area_spin.setValue(100); self.output_mode_combo.setCurrentText("binary_mask")
        self.log_text.append("SAM_HQ generator parameters reset to defaults.")

    def get_sam_params(self):
        return {"points_per_side": self.points_per_side_spin.value(), "points_per_batch": self.points_per_batch_spin.value(), "pred_iou_thresh": self.pred_iou_thresh_spin.value(), "stability_score_thresh": self.stability_score_thresh_spin.value(), "stability_score_offset": self.stability_score_offset_spin.value(), "box_nms_thresh": self.box_nms_thresh_spin.value(), "crop_n_layers": self.crop_n_layers_spin.value(), "crop_nms_thresh": self.crop_nms_thresh_spin.value(), "crop_overlap_ratio": self.crop_overlap_ratio_spin.value(), "crop_n_points_downscale_factor": self.crop_n_points_downscale_factor_spin.value(), "min_mask_region_area": self.min_mask_region_area_spin.value(), "output_mode": self.output_mode_combo.currentText()}

    def get_filter_params(self):
        return {"use_area": self.area_check.isChecked(), "use_intensity": self.intensity_check.isChecked(), "a_threshold": self.a_threshold_spin.value(), "i_threshold": self.i_threshold_spin.value(), "r_threshold": self.r_threshold_spin.value(), "min_aggregate": self.min_aggregate_spin.value(), "max_aggregate": self.max_aggregate_spin.value()}

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder: self.input_folder = folder; self.input_folder_label.setText(f"Input Folder: ...{folder[-40:]}"); self.update_button_states()

    def select_model_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select SAM Model File", "", "Model Files (*.pth)")
        if file: self.model_path = file; self.model_path_label.setText(f"Model File: {os.path.basename(file)}"); self.update_button_states()

    def update_button_states(self):
        input_ready = bool(self.input_folder); model_ready = bool(self.model_path)
        can_process = input_ready and model_ready and SEGMENT_ANYTHING_AVAILABLE
        self.process_btn.setEnabled(can_process)
        self.open_output_btn.setEnabled(input_ready)
        if not SEGMENT_ANYTHING_AVAILABLE:
            if not hasattr(self, "_lib_warning_shown") or not self._lib_warning_shown:
                self.log_text.append("Error: 'segment_anything' library not found or failed to import. Processing is disabled."); self._lib_warning_shown = True

    def open_output_folder(self):
        if not self.input_folder:
            QMessageBox.warning(self, "Input Missing", "Please select an input folder first.")
            return
        output_folder_path = self.get_processing_output_folder_path()
        if not output_folder_path:
            QMessageBox.warning(self, "Error", "Could not determine the output folder path.")
            return

        try:
            os.makedirs(output_folder_path, exist_ok=True)
            self.log_text.append(f"Ensuring output folder exists: {output_folder_path}")

            if sys.platform == "win32":
                os.startfile(output_folder_path)
            elif sys.platform == "darwin":
                os.system(f"open \"{output_folder_path}\"")
            else:
                os.system(f"xdg-open \"{output_folder_path}\"")
        except OSError as e:
             self.log_updated.emit(f"Error: Could not create or access output folder: {e}")
             QMessageBox.warning(self, "Creation/Access Failed", f"Could not create or access the output folder:\n{output_folder_path}\nError: {e}")
        except Exception as e:
            self.log_updated.emit(f"Error: Could not open output folder: {e}")
            QMessageBox.warning(self, "Open Failed", f"Could not open the output folder:\n{output_folder_path}\nError: {e}")

    def start_processing(self):
        min_agg = self.min_aggregate_spin.value(); max_agg = self.max_aggregate_spin.value()
        if min_agg > max_agg: QMessageBox.warning(self, "Invalid Thresholds", "Min Aggregate Threshold cannot be greater than Max Aggregate Threshold."); return
        if not self.area_check.isChecked() and not self.intensity_check.isChecked(): QMessageBox.warning(self, "Selection Required", "Please select at least one Phase 1 filter method."); return
        if not self.input_folder or not self.model_path: QMessageBox.warning(self, "Input Missing", "Please select both an input folder and a SAM model file."); return
        if not SEGMENT_ANYTHING_AVAILABLE: QMessageBox.critical(self, "Library Missing", "'segment_anything' library is unavailable."); return
        output_folder = self.get_processing_output_folder_path()
        if not output_folder: self.log_text.append("Error: Could not determine output folder path."); return

        self.log_text.append("-" * 20 + " New SAM_HQ Processing Task " + "-" * 20)
        self.log_text.append("Starting SAM_HQ image processing and filtering...")
        self.process_btn.setEnabled(False); self.stop_btn.setEnabled(True)

        sam_params = self.get_sam_params(); filter_params = self.get_filter_params()
        if self.process_thread and self.process_thread.isRunning(): self.log_text.append("Stopping previous processing task..."); self.process_thread.stop(); self.process_thread.wait(); self.log_text.append("Previous task stopped.")
        self.process_thread = HQSAMProcessorThread(self.input_folder, output_folder, self.model_path, filter_params, sam_params)
        self.process_thread.processing_finished.connect(self.on_processing_finished)
        self.process_thread.log_updated.connect(self.update_log)
        self.process_thread.start()

    def stop_processing(self):
        if self.process_thread and self.process_thread.isRunning(): self.process_thread.stop(); self.stop_btn.setEnabled(False)
        else: self.log_text.append("No processing task is currently running.")

    def update_log(self, message):
        self.log_text.append(message); scrollbar = self.log_text.verticalScrollBar(); scrollbar.setValue(scrollbar.maximum())

    def on_processing_finished(self):
        self.log_text.append("All image processing finished.")
        self.update_button_states(); self.stop_btn.setEnabled(False)
        self.open_output_btn.setEnabled(bool(self.input_folder))
        self.process_thread = None

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit', "Are you sure you want to exit?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if self.process_thread and self.process_thread.isRunning(): self.process_thread.stop(); self.process_thread.wait(3000)
            event.accept()
        else: event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if not SEGMENT_ANYTHING_AVAILABLE:
        msg_box = QMessageBox(); msg_box.setIcon(QMessageBox.Icon.Critical); msg_box.setWindowTitle("Dependency Missing")
        msg_box.setText("The 'segment_anything' library is required to run this program.\nPlease ensure it and its dependencies (like 'timm') are installed correctly according to the setup guide.")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok); msg_box.exec(); sys.exit(1)
    window = HQSAMMainWindow(); window.show(); sys.exit(app.exec())
