import hou
import torch
import torch.nn as nn
from PySide2 import QtWidgets, QtCore, QtGui
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    FluxPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    T2IAdapter  # Correctly imported from diffusers
)
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download, login
import os
import datetime
import io
import time
import json
import shutil
import logging

try:
    import pynvml
    pynvml.nvmlInit()
except:
    pynvml = None

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_lora_weights(pipe, lora_path, lora_scale=1.0):
    """Load LoRA weights into the pipeline"""
    try:
        logger.debug(f"Loading LoRA weights from {lora_path}")
        pipe.load_lora_weights(lora_path)
        return pipe, lora_scale
    except Exception as e:
        logger.error(f"Error loading LoRA weights: {str(e)}")
        raise

class ModelManager(QtWidgets.QDialog):
    def __init__(self, parent=None, huggingface_token=None):
        super().__init__(parent)
        self.huggingface_token = huggingface_token
        self.api = HfApi()
        self.setWindowTitle("Model Manager")
        self.initUI()
        
    def initUI(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Available Models:"))
        self.model_list = QtWidgets.QListWidget()
        self.refresh_model_list()
        layout.addWidget(self.model_list)
        button_layout = QtWidgets.QHBoxLayout()
        self.download_btn = QtWidgets.QPushButton("Download")
        self.download_btn.clicked.connect(self.download_model)
        self.delete_btn = QtWidgets.QPushButton("Delete Local")
        self.delete_btn.clicked.connect(self.delete_model)
        self.refresh_btn = QtWidgets.QPushButton("Refresh List")
        self.refresh_btn.clicked.connect(self.refresh_model_list)
        button_layout.addWidget(self.download_btn)
        button_layout.addWidget(self.delete_btn)
        button_layout.addWidget(self.refresh_btn)
        layout.addLayout(button_layout)
        self.status_label = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status_label)
        
    def refresh_model_list(self):
        self.model_list.clear()
        try:
            models = self.api.list_models(author="stabilityai", sort="downloads", direction=-1, limit=20)
            for model in models:
                item = QtWidgets.QListWidgetItem(model.modelId)
                local_path = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), f"models--{model.modelId.replace('/', '--')}")
                if os.path.exists(local_path):
                    item.setForeground(QtGui.QBrush(QtGui.QColor("green")))
                self.model_list.addItem(item)
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            
    def download_model(self):
        selected = self.model_list.currentItem()
        if selected:
            model_id = selected.text()
            self.status_label.setText(f"Downloading {model_id}...")
            try:
                hf_hub_download(repo_id=model_id, filename="model_index.json", token=self.huggingface_token)
                self.status_label.setText(f"Downloaded {model_id}")
                self.refresh_model_list()
            except Exception as e:
                self.status_label.setText(f"Download error: {str(e)}")
                
    def delete_model(self):
        selected = self.model_list.currentItem()
        if selected:
            model_id = selected.text()
            local_path = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), f"models--{model_id.replace('/', '--')}")
            if os.path.exists(local_path):
                try:
                    shutil.rmtree(local_path)
                    self.status_label.setText(f"Deleted {model_id}")
                    self.refresh_model_list()
                except Exception as e:
                    self.status_label.setText(f"Delete error: {str(e)}")

class ImagePreviewLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(256, 256)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText("No image")
        self.setStyleSheet("QLabel { background-color: #2a2a2a; border: 1px solid #444; }")
        self._pixmap = None
    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            super().setPixmap(scaled)
        else:
            self.setText("No image")
    def resizeEvent(self, event):
        if self._pixmap:
            scaled = self._pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            super().setPixmap(scaled)
        super().resizeEvent(event)

class RenderWindow(QtWidgets.QMainWindow):
    def __init__(self, pixmap):
        super().__init__()
        self.setWindowTitle("Generated Render")
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setPixmap(pixmap)
        self.setCentralWidget(self.label)
        self.resize(pixmap.size())

class SDXLPanel(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int)
    status_signal = QtCore.Signal(str)
    enable_render_signal = QtCore.Signal(bool)
    enable_cancel_signal = QtCore.Signal(bool)
    show_progress_signal = QtCore.Signal(bool)
    input_preview_signal = QtCore.Signal(QtGui.QPixmap)
    output_preview_signal = QtCore.Signal(QtGui.QPixmap)
    display_signal = QtCore.Signal(QtGui.QPixmap)

    def __init__(self):
        super().__init__()
        self.cancel_flag = False
        self.adapter_checkboxes = {}
        self.huggingface_token = ""
        self.display_window = None
        self.cached_pipe = None
        self.cached_model_name = ""
        self.cached_adapters = []
        self.cached_lora_path = ""
        self.cached_lora_scale = 1.0
        self.cached_ti_token = ""
        self.history = []
        self.gpu_handle = None
        self.use_cpu_only = False
        self.initUI()
        self.setup_signals()
        self.setup_performance_monitoring()
        logger.info("SDXLPanel initialized")

    def setup_signals(self):
        self.progress_signal.connect(self.update_progress_bar)
        self.status_signal.connect(self.update_status)
        self.enable_render_signal.connect(self.set_render_enabled)
        self.enable_cancel_signal.connect(self.set_cancel_enabled)
        self.show_progress_signal.connect(self.set_progress_visible)
        self.input_preview_signal.connect(self.update_input_preview)
        self.output_preview_signal.connect(self.update_output_preview)
        self.display_signal.connect(self._show_display)

    def initUI(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        tab_widget = QtWidgets.QTabWidget()
        
        generation_tab = QtWidgets.QWidget()
        gen_layout = QtWidgets.QVBoxLayout(generation_tab)
        
        top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        prompt_widget = QtWidgets.QWidget()
        prompt_layout = QtWidgets.QVBoxLayout(prompt_widget)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        
        prompt_layout.addWidget(QtWidgets.QLabel("Prompt:"))
        self.prompt_input = QtWidgets.QTextEdit()
        self.prompt_input.setPlaceholderText("A photorealistic render of a scene, high detail, studio lighting, PBR materials")
        self.prompt_input.setMaximumHeight(80)
        prompt_layout.addWidget(self.prompt_input)
        
        prompt_layout.addWidget(QtWidgets.QLabel("Negative Prompt:"))
        self.negative_prompt = QtWidgets.QTextEdit()
        self.negative_prompt.setPlaceholderText("blurry, low detail, cartoon, sketch, painting, unrealistic, oversaturated")
        self.negative_prompt.setMaximumHeight(80)
        prompt_layout.addWidget(self.negative_prompt)
        
        self.caption_btn = QtWidgets.QPushButton("Generate Caption from Viewport")
        self.caption_btn.clicked.connect(self.generate_caption)
        prompt_layout.addWidget(self.caption_btn)
        
        top_splitter.addWidget(prompt_widget)
        
        preview_widget = QtWidgets.QWidget()
        preview_layout = QtWidgets.QHBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        input_preview_group = QtWidgets.QGroupBox("Input")
        input_preview_layout = QtWidgets.QVBoxLayout(input_preview_group)
        self.input_preview = ImagePreviewLabel()
        self.input_preview.setMinimumSize(200, 200)
        input_preview_layout.addWidget(self.input_preview)
        
        output_preview_group = QtWidgets.QGroupBox("Output")
        output_preview_layout = QtWidgets.QVBoxLayout(output_preview_group)
        self.output_preview = ImagePreviewLabel()
        self.output_preview.setMinimumSize(200, 200)
        output_preview_layout.addWidget(self.output_preview)
        
        preview_layout.addWidget(input_preview_group)
        preview_layout.addWidget(output_preview_group)
        
        top_splitter.addWidget(preview_widget)
        top_splitter.setSizes([300, 300])
        gen_layout.addWidget(top_splitter)
        
        params_widget = QtWidgets.QWidget()
        params_layout = QtWidgets.QGridLayout(params_widget)
        params_layout.setContentsMargins(0, 0, 0, 0)
        
        row = 0
        params_layout.addWidget(QtWidgets.QLabel("Base Model:"), row, 0)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems([
            "stabilityai/stable-diffusion-xl-base-1.0",
            "dreamlike-art/dreamlike-photoreal-2.0",
            "lykon/dreamshaper-xl-lightning",
            "Lykon/dreamshaper-xl-v2-turbo",
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "TencentARC/flux-mini"  # Replaced FLUX.1-dev with flux-mini
        ])
        params_layout.addWidget(self.model_combo, row, 1)
        
        params_layout.addWidget(QtWidgets.QLabel("Scheduler:"), row, 2)
        self.scheduler_combo = QtWidgets.QComboBox()
        self.scheduler_combo.addItems([
            "DPM++ SDE Karras",
            "UniPC",
            "DPM++ 2M Karras",
            "DPM++ 2M",
            "Euler",
            "Euler Ancestral",
            "DDIM"
        ])
        params_layout.addWidget(self.scheduler_combo, row, 3)
        
        row = 1
        params_layout.addWidget(QtWidgets.QLabel("Dimensions:"), row, 0)
        dim_widget = QtWidgets.QWidget()
        dim_layout = QtWidgets.QHBoxLayout(dim_widget)
        dim_layout.setContentsMargins(0, 0, 0, 0)
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(512, 2048)
        self.width_spin.setValue(512)
        self.width_spin.setSingleStep(64)
        self.height_spin = QtWidgets.QSpinBox()
        self.height_spin.setRange(512, 2048)
        self.height_spin.setValue(512)
        self.height_spin.setSingleStep(64)
        dim_layout.addWidget(QtWidgets.QLabel("W:"))
        dim_layout.addWidget(self.width_spin)
        dim_layout.addWidget(QtWidgets.QLabel("H:"))
        dim_layout.addWidget(self.height_spin)
        params_layout.addWidget(dim_widget, row, 1)
        
        params_layout.addWidget(QtWidgets.QLabel("Seed:"), row, 2)
        self.seed_input = QtWidgets.QSpinBox()
        self.seed_input.setRange(-1, 2147483647)
        self.seed_input.setValue(-1)
        self.seed_input.setSpecialValueText("Random")
        params_layout.addWidget(self.seed_input, row, 3)
        
        row = 2
        params_layout.addWidget(QtWidgets.QLabel("Guidance Scale:"), row, 0)
        self.guidance_scale = QtWidgets.QDoubleSpinBox()
        self.guidance_scale.setRange(1.0, 20.0)
        self.guidance_scale.setValue(7.5)
        self.guidance_scale.setSingleStep(0.5)
        params_layout.addWidget(self.guidance_scale, row, 1)
        
        params_layout.addWidget(QtWidgets.QLabel("Steps:"), row, 2)
        self.num_steps = QtWidgets.QSpinBox()
        self.num_steps.setRange(1, 100)
        self.num_steps.setValue(20)
        params_layout.addWidget(self.num_steps, row, 3)
        
        row = 3
        params_layout.addWidget(QtWidgets.QLabel("Strength:"), row, 0)
        self.strength = QtWidgets.QDoubleSpinBox()
        self.strength.setRange(0.0, 1.0)
        self.strength.setValue(0.4)
        self.strength.setSingleStep(0.1)
        params_layout.addWidget(self.strength, row, 1)
        
        params_layout.addWidget(QtWidgets.QLabel("Output Dir:"), row, 2)
        dir_widget = QtWidgets.QWidget()
        dir_layout = QtWidgets.QHBoxLayout(dir_widget)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        self.output_dir = QtWidgets.QLineEdit()
        self.output_dir.setText(hou.expandString("$HIP/render"))
        browse_btn = QtWidgets.QPushButton("...")
        browse_btn.setMaximumWidth(30)
        browse_btn.clicked.connect(self.browse_output)
        dir_layout.addWidget(self.output_dir)
        dir_layout.addWidget(browse_btn)
        params_layout.addWidget(dir_widget, row, 3)
        
        gen_layout.addWidget(params_widget)
        
        control_group = QtWidgets.QGroupBox("Control Models (Select only one)")
        control_group.setCheckable(True)
        control_group.setChecked(False)
        control_layout = QtWidgets.QVBoxLayout(control_group)
        
        control_models = [
            "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
            "TencentARC/t2i-adapter-canny-sdxl-1.0",
            "TencentARC/t2i-adapter-sketch-sdxl-1.0",
            "TencentARC/t2i-adapter-seg-sdxl-1.0",
            "TencentARC/t2i-adapter-lineart-sdxl-1.0",
            "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"  # Kept for compatibility, may need adjustment
        ]
        
        control_grid = QtWidgets.QGridLayout()
        row = 0
        col = 0
        self.control_group = QtWidgets.QButtonGroup()
        self.control_group.setExclusive(True)
        for model in control_models:
            short_name = model.split('/')[-1].replace('t2i-adapter-', '').replace('-sdxl-1.0', '').replace('FLUX.1-dev-', '')
            checkbox = QtWidgets.QCheckBox(short_name)
            self.adapter_checkboxes[model] = checkbox
            self.control_group.addButton(checkbox)
            control_grid.addWidget(checkbox, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        control_layout.addLayout(control_grid)
        
        scale_layout = QtWidgets.QHBoxLayout()
        scale_layout.addWidget(QtWidgets.QLabel("Control Scale:"))
        self.adapter_scale = QtWidgets.QDoubleSpinBox()
        self.adapter_scale.setRange(0.0, 2.0)
        self.adapter_scale.setValue(1.0)
        self.adapter_scale.setSingleStep(0.1)
        scale_layout.addWidget(self.adapter_scale)
        scale_layout.addStretch()
        
        control_layout.addLayout(scale_layout)
        self.keep_models_loaded_checkbox = QtWidgets.QCheckBox("Keep models loaded in GPU")
        self.keep_models_loaded_checkbox.setChecked(True)
        control_layout.addWidget(self.keep_models_loaded_checkbox)
        
        gen_layout.addWidget(control_group)
        
        self.lora_group = QtWidgets.QGroupBox("LoRA Settings")
        self.lora_group.setCheckable(True)
        self.lora_group.setChecked(False)
        lora_layout = QtWidgets.QVBoxLayout(self.lora_group)
        
        lora_file_layout = QtWidgets.QHBoxLayout()
        lora_file_layout.addWidget(QtWidgets.QLabel("LoRA Model:"))
        self.lora_path = QtWidgets.QLineEdit()
        self.lora_path.setPlaceholderText("Path to LoRA weights")
        lora_file_layout.addWidget(self.lora_path)
        lora_browse_btn = QtWidgets.QPushButton("Browse")
        lora_browse_btn.clicked.connect(self.browse_lora)
        lora_file_layout.addWidget(lora_browse_btn)
        lora_layout.addLayout(lora_file_layout)
        
        lora_scale_layout = QtWidgets.QHBoxLayout()
        lora_scale_layout.addWidget(QtWidgets.QLabel("LoRA Scale:"))
        self.lora_scale = QtWidgets.QDoubleSpinBox()
        self.lora_scale.setRange(0.0, 2.0)
        self.lora_scale.setValue(0.7)
        self.lora_scale.setSingleStep(0.1)
        lora_scale_layout.addWidget(self.lora_scale)
        lora_layout.addLayout(lora_scale_layout)
        
        ti_layout = QtWidgets.QHBoxLayout()
        ti_layout.addWidget(QtWidgets.QLabel("Textual Inversion Token:"))
        self.ti_token = QtWidgets.QLineEdit()
        self.ti_token.setPlaceholderText("Enter custom token (e.g., <photorealistic-render>)")
        ti_layout.addWidget(self.ti_token)
        lora_layout.addLayout(ti_layout)
        
        gen_layout.addWidget(self.lora_group)
        
        bottom_layout = QtWidgets.QHBoxLayout()
        
        status_layout = QtWidgets.QVBoxLayout()
        self.status_label = QtWidgets.QLabel("Ready")
        status_layout.addWidget(self.status_label)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        bottom_layout.addLayout(status_layout, 2)
        
        button_layout = QtWidgets.QHBoxLayout()
        self.render_btn = QtWidgets.QPushButton("Generate")
        self.render_btn.clicked.connect(self.start_generation)
        self.render_btn.setMinimumWidth(100)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_generation)
        self.cancel_btn.setEnabled(False)
        self.cpu_only_btn = QtWidgets.QPushButton("Toggle CPU Only")
        self.cpu_only_btn.clicked.connect(self.toggle_cpu_only)
        button_layout.addWidget(self.render_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.cpu_only_btn)
        bottom_layout.addLayout(button_layout, 1)
        
        gen_layout.addLayout(bottom_layout)

        perf_layout = QtWidgets.QHBoxLayout()
        self.performance_label = QtWidgets.QLabel("GPU: N/A")
        perf_layout.addWidget(self.performance_label)
        gen_layout.addLayout(perf_layout)
        
        presets_tab = QtWidgets.QWidget()
        presets_layout = QtWidgets.QVBoxLayout(presets_tab)
        
        preset_layout = QtWidgets.QHBoxLayout()
        preset_layout.addWidget(QtWidgets.QLabel("Presets:"))
        self.save_preset_btn = QtWidgets.QPushButton("Save")
        self.save_preset_btn.clicked.connect(self.save_preset)
        self.load_preset_btn = QtWidgets.QPushButton("Load")
        self.load_preset_btn.clicked.connect(self.load_preset)
        self.renderer_preset_btn = QtWidgets.QPushButton("Load Renderer Preset")
        self.renderer_preset_btn.clicked.connect(self.load_renderer_preset)
        preset_layout.addWidget(self.save_preset_btn)
        preset_layout.addWidget(self.load_preset_btn)
        preset_layout.addWidget(self.renderer_preset_btn)
        preset_layout.addStretch()
        
        presets_layout.addLayout(preset_layout)
        
        presets_layout.addWidget(QtWidgets.QLabel("Generation History:"))
        self.history_list = QtWidgets.QListWidget()
        presets_layout.addWidget(self.history_list)
        
        settings_layout = QtWidgets.QHBoxLayout()
        settings_layout.addStretch()
        self.settings_btn = QtWidgets.QPushButton("HF Token Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        self.model_manager_btn = QtWidgets.QPushButton("Model Manager")
        self.model_manager_btn.clicked.connect(self.open_model_manager)
        settings_layout.addWidget(self.settings_btn)
        settings_layout.addWidget(self.model_manager_btn)
        presets_layout.addLayout(settings_layout)
        
        tab_widget.addTab(generation_tab, "Generate")
        tab_widget.addTab(presets_tab, "Presets & History")
        main_layout.addWidget(tab_widget)

    def open_settings(self):
        token, ok = QtWidgets.QInputDialog.getText(self, "Settings", "Hugging Face Token:")
        if ok and token:
            self.huggingface_token = token
            login(token=self.huggingface_token)
            self.update_status("Settings updated and logged in to Hugging Face.")

    def open_model_manager(self):
        manager = ModelManager(self, self.huggingface_token)
        manager.exec_()

    def browse_output(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir.text())
        if directory:
            self.output_dir.setText(directory)

    def browse_lora(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select LoRA Weights", "", "LoRA Files (*.safetensors *.pt *.bin)")
        if file_path:
            self.lora_path.setText(file_path)
            logger.debug(f"Selected LoRA file: {file_path}")

    def toggle_cpu_only(self):
        self.use_cpu_only = not self.use_cpu_only
        self.update_status(f"CPU-only mode: {'Enabled' if self.use_cpu_only else 'Disabled'}")
        logger.info(f"CPU-only mode toggled to: {self.use_cpu_only}")

    def get_selected_adapter_models(self):
        selected = [model for model, checkbox in self.adapter_checkboxes.items() if checkbox.isChecked()]
        if len(selected) > 1:
            raise ValueError("Only one control model can be selected at a time.")
        return selected

    def generate_caption(self):
        output_dir = self.output_dir.text()
        image = self.capture_viewport(output_dir)
        if image is None:
            self.update_status("Failed to capture viewport for captioning")
            return
        if image.mode != "RGB":
            image = image.convert("RGB")
        if not hasattr(self, 'caption_model'):
            self.caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to("cuda" if not self.use_cpu_only else "cpu")
            self.caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        inputs = self.caption_processor(images=image, return_tensors="pt").to("cuda" if not self.use_cpu_only else "cpu")
        output_ids = self.caption_model.generate(**inputs, max_length=16)
        caption = self.caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_text = f"A photorealistic render of {caption}, high detail, studio lighting, PBR materials"
        self.prompt_input.setPlainText(new_text)
        self.update_status("Caption generated and added to prompt")

    def progress_callback(self, step, timestep, latents):
        total_steps = self.num_steps.value()
        progress = int((step / total_steps) * 100)
        self.progress_signal.emit(progress)

    def generate_image(self):
        start_time = time.time()
        pipe = None
        lora_scale = 1.0
        if not hasattr(self, 'cached_pipe'):
            self.cached_pipe = None
            self.cached_model_name = ""
            self.cached_adapters = []
            self.cached_lora_path = ""
            self.cached_lora_scale = 1.0
            self.cached_ti_token = ""
        
        try:
            logger.debug("Starting image generation")
            torch.cuda.empty_cache()
            prep_start = time.time()
            output_dir = self.output_dir.text()
            os.makedirs(output_dir, exist_ok=True)
            selected_models = self.get_selected_adapter_models()
            lora_enabled = self.lora_group.isChecked() and self.lora_path.text()
            ti_enabled = self.lora_group.isChecked() and self.ti_token.text()
            cond_img = None
            if selected_models:
                self.update_status("Capturing viewport...")
                cond_img = self.capture_viewport(output_dir)
                if cond_img is None:
                    self.update_status("Failed to capture viewport!")
                    return
                pixmap = self.pil_image_to_pixmap(cond_img)
                self.input_preview_signal.emit(pixmap)
            prep_time = time.time() - prep_start
            logger.debug(f"Preparation time: {prep_time:.2f} seconds")
    
            model_start = time.time()
            is_flux_model = self.model_combo.currentText() == "TencentARC/flux-mini"
            reuse_model = (self.keep_models_loaded_checkbox.isChecked() and 
                           self.cached_pipe is not None and 
                           self.cached_model_name == self.model_combo.currentText() and
                           self.cached_adapters == selected_models and
                           self.cached_lora_path == (self.lora_path.text() if lora_enabled else "") and
                           self.cached_lora_scale == (self.lora_scale.value() if lora_enabled else 1.0) and
                           self.cached_ti_token == (self.ti_token.text() if ti_enabled else ""))
            
            logger.debug(f"Reuse model check: {reuse_model}")
            
            if not reuse_model:
                self.update_status("Loading new model...")
                logger.debug(f"Loading model {self.model_combo.currentText()}")
                scheduler_name = self.scheduler_combo.currentText()
                scheduler_config = self.get_scheduler_config(scheduler_name)
                hf_token = self.huggingface_token if self.huggingface_token else None
                torch_dtype = torch.bfloat16 if is_flux_model else torch.float16
                device = "cpu" if self.use_cpu_only else "cuda"
                
                from diffusers import DiffusionPipeline, FluxTransformer2DModel, AutoencoderKL
                from transformers import CLIPTokenizer
                
                if is_flux_model:
                    try:
                        pipe = DiffusionPipeline.from_pretrained(
                            "TencentARC/flux-mini",
                            torch_dtype=torch_dtype,
                            use_auth_token=hf_token
                        ).to(device)
                        logger.debug("Loaded TencentARC/flux-mini with DiffusionPipeline")
                    except Exception as e:
                        logger.warning(f"DiffusionPipeline failed: {str(e)}. Falling back to manual component loading.")
                        self.update_status("Warning: Falling back to manual component loading for flux-mini.")
                        
                        # Manual component loading for flux-mini
                        # Load config from FLUX.1-dev since flux-mini lacks it
                        transformer_config = FluxTransformer2DModel.from_pretrained(
                            "black-forest-labs/FLUX.1-dev",
                            subfolder="transformer",
                            torch_dtype=torch_dtype,
                            use_auth_token=hf_token,
                            only_config=True  # Hypothetical argument; we'll load weights separately
                        ).config
                        transformer = FluxTransformer2DModel.from_config(transformer_config).to(device)
                        # Load weights from flux-mini
                        transformer.load_state_dict(
                            torch.load(
                                hf_hub_download(
                                    "TencentARC/flux-mini",
                                    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                                    token=hf_token
                                ),
                                map_location=device
                            ),
                            strict=False
                        )
                        transformer.load_state_dict(
                            torch.load(
                                hf_hub_download(
                                    "TencentARC/flux-mini",
                                    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
                                    token=hf_token
                                ),
                                map_location=device
                            ),
                            strict=False
                        )
                        vae = AutoencoderKL.from_pretrained(
                            "black-forest-labs/FLUX.1-dev",
                            subfolder="vae",
                            torch_dtype=torch_dtype,
                            use_auth_token=hf_token
                        ).to(device)
                        tokenizer = CLIPTokenizer.from_pretrained(
                            "black-forest-labs/FLUX.1-dev",
                            subfolder="tokenizer",
                            use_auth_token=hf_token
                        )
                        pipe = FluxPipeline(
                            transformer=transformer,
                            vae=vae,
                            tokenizer=tokenizer,
                            scheduler=scheduler_config,
                        ).to(device)
                        logger.debug("Manually loaded flux-mini components with weights from TencentARC/flux-mini")
                    
                    if selected_models and selected_models[0] == "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro":
                        self.update_status("Warning: ControlNet compatibility with flux-mini unverified.")
                        pipe.load_lora_weights("Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro", use_auth_token=hf_token)
                else:
                    if selected_models and selected_models[0] != "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro":
                        adapter = T2IAdapter.from_pretrained(
                            selected_models[0],
                            torch_dtype=torch_dtype,
                            variant="fp16",
                            use_auth_token=hf_token
                        ).to(device)
                        for param in adapter.parameters():
                            param.requires_grad = False
                        pipe = StableDiffusionXLPipeline.from_pretrained(
                            self.model_combo.currentText(),
                            adapter=adapter,
                            torch_dtype=torch_dtype,
                            variant="fp16",
                            use_safetensors=True,
                            use_auth_token=hf_token
                        ).to(device)
                    else:
                        pipe = StableDiffusionXLPipeline.from_pretrained(
                            self.model_combo.currentText(),
                            torch_dtype=torch_dtype,
                            variant="fp16",
                            use_safetensors=True,
                            use_auth_token=hf_token
                        ).to(device)
                
                if lora_enabled and (not is_flux_model or selected_models[0] != "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"):
                    self.update_status("Loading LoRA weights...")
                    pipe, lora_scale = load_lora_weights(pipe, self.lora_path.text(), self.lora_scale.value())
                
                if ti_enabled and not is_flux_model:
                    ti_token = self.ti_token.text()
                    self.update_status("Applying Textual Inversion...")
                    logger.debug(f"Applying Textual Inversion with token {ti_token}")
                    if not os.path.isfile(ti_token):
                        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
                        if (not ti_token or any(c not in valid_chars for c in ti_token) or 
                            ti_token.startswith(('-', '.')) or ti_token.endswith(('-', '.')) or len(ti_token) > 96):
                            raise ValueError(f"Invalid Textual Inversion token '{ti_token}'.")
                    pipe.load_textual_inversion(ti_token)
                
                if not is_flux_model:
                    pipe.scheduler = scheduler_config.from_config(pipe.scheduler.config)
                pipe = optimize_pipeline(pipe)
                
                if self.keep_models_loaded_checkbox.isChecked():
                    self.cached_pipe = pipe
                    self.cached_model_name = self.model_combo.currentText()
                    self.cached_adapters = selected_models
                    self.cached_lora_path = self.lora_path.text() if lora_enabled and not is_flux_model else ""
                    self.cached_lora_scale = self.lora_scale.value() if lora_enabled and not is_flux_model else 1.0
                    self.cached_ti_token = self.ti_token.text() if ti_enabled and not is_flux_model else ""
                    logger.debug(f"Cached model {self.cached_model_name}")
            else:
                self.update_status("Reusing loaded model...")
                pipe = self.cached_pipe
                scheduler_name = self.scheduler_combo.currentText()
                scheduler_config = self.get_scheduler_config(scheduler_name)
                pipe.scheduler = scheduler_config.from_config(pipe.scheduler.config)
                lora_scale = self.cached_lora_scale
            model_time = time.time() - model_start
            logger.debug(f"Model loading/checking time: {model_time:.2f} seconds")
    
            gen_start = time.time()
            use_cuda_graph = not selected_models and not self.use_cpu_only
            if is_flux_model:
                generation_args = {
                    "prompt": self.prompt_input.toPlainText(),
                    "height": self.height_spin.value(),
                    "width": self.width_spin.value(),
                    "guidance_scale": self.guidance_scale.value() if self.guidance_scale.value() != 7.5 else 3.5,
                    "num_inference_steps": self.num_steps.value() if self.num_steps.value() != 20 else 25,
                    "max_sequence_length": 512,
                }
                if self.seed_input.value() != -1:
                    generation_args["generator"] = torch.Generator("cuda" if not self.use_cpu_only else "cpu").manual_seed(self.seed_input.value())
            else:
                generation_args = {
                    "prompt": self.prompt_input.toPlainText(),
                    "negative_prompt": self.negative_prompt.toPlainText(),
                    "height": self.height_spin.value(),
                    "width": self.width_spin.value(),
                    "guidance_scale": self.guidance_scale.value(),
                    "num_inference_steps": self.num_steps.value(),
                    "callback": self.progress_callback,
                    "callback_steps": 1,
                }
                if ti_enabled:
                    generation_args["prompt"] = f"{self.ti_token.text()} {generation_args['prompt']}"
                if lora_enabled:
                    generation_args["cross_attention_kwargs"] = {"scale": lora_scale}
                if self.model_combo.currentText() == "lykon/dreamshaper-xl-lightning":
                    generation_args["guidance_scale"] = 2
                    generation_args["num_inference_steps"] = 4
                else:
                    generation_args["strength"] = self.strength.value() if cond_img else 1.0
                if self.seed_input.value() != -1:
                    generation_args["generator"] = torch.Generator("cuda" if not self.use_cpu_only else "cpu").manual_seed(self.seed_input.value())
                if selected_models and selected_models[0] != "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro":
                    generation_args["adapter_conditioning_scale"] = self.adapter_scale.value()
                    generation_args["image"] = cond_img
            
            logger.debug(f"Generating with args: {generation_args}")
            if use_cuda_graph and hasattr(pipe, 'enable_sequential_cpu_offload'):
                with torch.cuda.amp.autocast():
                    result = pipe(**generation_args)
            else:
                with torch.inference_mode():
                    result = pipe(**generation_args)
            gen_time = time.time() - gen_start
            logger.debug(f"Generation time: {gen_time:.2f} seconds")
    
            post_start = time.time()
            elapsed_time = time.time() - start_time
            if not self.cancel_flag and result.images:
                final_image = result.images[0]
                pixmap = self.pil_image_to_pixmap(final_image)
                self.output_preview_signal.emit(pixmap)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_dir, f"generated_{timestamp}.png")
                final_image.save(output_path)
                seed_info = f" (Seed: {self.seed_input.value()})" if self.seed_input.value() != -1 else ""
                self.update_status(f"Image saved: {output_path}{seed_info} | Generation took {elapsed_time:.2f} seconds")
                self.display_signal.emit(pixmap)
                history_entry = f"{timestamp} - {output_path} - Seed: {self.seed_input.value()}"
                self.history.append(history_entry)
                self.history_list.addItem(history_entry)
            post_time = time.time() - post_start
            logger.debug(f"Post-processing time: {post_time:.2f} seconds")
        
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory: {str(e)}")
            self.update_status("GPU memory exceeded. Try lower resolution, fewer steps, or CPU-only mode.")
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            self.update_status(f"Error: {str(e)}")
        finally:
            if not self.keep_models_loaded_checkbox.isChecked() and pipe is not None:
                del pipe
                self.cached_pipe = None
                self.cached_model_name = ""
                self.cached_adapters = []
                self.cached_lora_path = ""
                self.cached_lora_scale = 1.0
                self.cached_ti_token = ""
                torch.cuda.empty_cache()
            self.enable_render_signal.emit(True)
            self.enable_cancel_signal.emit(False)
            self.show_progress_signal.emit(False)



    def _show_display(self, pixmap):
        display_window = QtWidgets.QMainWindow()
        display_window.setWindowTitle("Generated Display")
        label = QtWidgets.QLabel()
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setPixmap(pixmap)
        display_window.setCentralWidget(label)
        display_window.resize(pixmap.size())
        display_window.show()
        self.display_window = display_window

    def update_input_preview(self, pixmap):
        self.input_preview.setPixmap(pixmap)

    def update_output_preview(self, pixmap):
        self.output_preview.setPixmap(pixmap)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, text):
        self.status_label.setText(text)
        logger.info(f"Status updated: {text}")

    def set_render_enabled(self, enabled):
        self.render_btn.setEnabled(enabled)

    def set_cancel_enabled(self, enabled):
        self.cancel_btn.setEnabled(enabled)

    def set_progress_visible(self, visible):
        self.progress_bar.setVisible(visible)

    def pil_image_to_pixmap(self, pil_image):
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        data = pil_image.tobytes("raw", "RGB")
        qimage = QtGui.QImage(data, pil_image.size[0], pil_image.size[1], QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimage)

    def start_generation(self):
        torch.cuda.empty_cache()
        self.cancel_flag = False
        self.enable_render_signal.emit(False)
        self.enable_cancel_signal.emit(True)
        self.show_progress_signal.emit(True)
        self.progress_signal.emit(0)
        self.update_status("Initializing...")
        QtCore.QTimer.singleShot(0, self.generate_image)

    def cancel_generation(self):
        self.cancel_flag = True
        self.update_status("Canceling...")

    def capture_viewport(self, output_dir):
        try:
            cur_desktop = hou.ui.curDesktop()
            scene_viewer = cur_desktop.paneTabOfType(hou.paneTabType.SceneViewer)
            if scene_viewer is None:
                for pane_tab in cur_desktop.paneTabs():
                    if isinstance(pane_tab, hou.SceneViewer):
                        scene_viewer = pane_tab
                        break
            if scene_viewer is None:
                self.update_status("No scene viewer found!")
                return None
            viewport = scene_viewer.curViewport()
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(output_dir, f"viewport_{timestamp}.jpg")
            try:
                self.update_status("Capturing viewport using viewwrite...")
                camera_path = f"{cur_desktop.name()}.{scene_viewer.name()}.world.{viewport.name()}"
                frame = hou.frame()
                hou.hscript(f'viewwrite -r 512 512 -f {frame} {frame} {camera_path} "{filename}"')
            except:
                self.update_status("Trying alternate capture method...")
                viewport.saveFrame(filename)
            if os.path.exists(filename):
                self.update_status("Loading captured image...")
                image = Image.open(filename)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                if image.size[0] < 512 or image.size[1] < 512:
                    image = image.resize((512, 512), Image.Resampling.LANCZOS)
                return image
            else:
                self.update_status("Failed to save viewport image!")
                return None
        except Exception as e:
            logger.error(f"Viewport capture error: {str(e)}")
            self.update_status(f"Viewport capture error: {str(e)}")
            return None

    def get_scheduler_config(self, scheduler_name):
        scheduler_map = {
            "DPM++ SDE Karras": lambda: DPMSolverSDEScheduler(use_karras_sigmas=True),
            "DPM++ 2M": lambda: DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_order=2),
            "DPM++ 2M Karras": lambda: DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True),
            "Euler": lambda: EulerDiscreteScheduler(),
            "Euler Ancestral": lambda: EulerAncestralDiscreteScheduler(),
            "DDIM": lambda: DDIMScheduler(),
            "UniPC": lambda: UniPCMultistepScheduler()
        }
        return scheduler_map[scheduler_name]()

    def save_preset(self):
        preset = {
            "prompt": self.prompt_input.toPlainText(),
            "negative_prompt": self.negative_prompt.toPlainText(),
            "base_model": self.model_combo.currentText(),
            "adapter_models": self.get_selected_adapter_models(),
            "adapter_scale": self.adapter_scale.value(),
            "scheduler": self.scheduler_combo.currentText(),
            "output_dir": self.output_dir.text(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "strength": self.strength.value(),
            "guidance_scale": self.guidance_scale.value(),
            "num_steps": self.num_steps.value(),
            "seed": self.seed_input.value(),
            "keep_models_loaded": self.keep_models_loaded_checkbox.isChecked(),
            "lora_enabled": self.lora_group.isChecked(),
            "lora_path": self.lora_path.text(),
            "lora_scale": self.lora_scale.value(),
            "ti_token": self.ti_token.text()
        }
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset", "Preset Name:")
        if ok and name:
            presets_dir = os.path.join(os.getcwd(), "presets")
            if not os.path.exists(presets_dir):
                os.makedirs(presets_dir)
            file_path = os.path.join(presets_dir, f"{name}.json")
            with open(file_path, "w") as f:
                json.dump(preset, f)
            self.update_status(f"Preset '{name}' saved.")

    def load_preset(self):
        presets_dir = os.path.join(os.getcwd(), "presets")
        if not os.path.exists(presets_dir):
            self.update_status("No presets found.")
            return
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Preset", presets_dir, "JSON Files (*.json)")
        if file_path:
            with open(file_path, "r") as f:
                preset = json.load(f)
            self.prompt_input.setPlainText(preset.get("prompt", ""))
            self.negative_prompt.setPlainText(preset.get("negative_prompt", ""))
            base_model = preset.get("base_model", "")
            index = self.model_combo.findText(base_model)
            if index != -1:
                self.model_combo.setCurrentIndex(index)
            adapter_models = preset.get("adapter_models", [])
            if len(adapter_models) > 1:
                self.update_status("Warning: Preset contains multiple adapters; only the first will be used.")
                adapter_models = [adapter_models[0]]
            for model, checkbox in self.adapter_checkboxes.items():
                checkbox.setChecked(model in adapter_models)
            self.adapter_scale.setValue(preset.get("adapter_scale", 1.0))
            scheduler = preset.get("scheduler", "")
            index = self.scheduler_combo.findText(scheduler)
            if index != -1:
                self.scheduler_combo.setCurrentIndex(index)
            self.output_dir.setText(preset.get("output_dir", hou.expandString("$HIP/render")))
            self.width_spin.setValue(preset.get("width", 512))
            self.height_spin.setValue(preset.get("height", 512))
            self.strength.setValue(preset.get("strength", 0.4))
            self.guidance_scale.setValue(preset.get("guidance_scale", 7.5))
            self.num_steps.setValue(preset.get("num_steps", 20))
            self.seed_input.setValue(preset.get("seed", -1))
            self.keep_models_loaded_checkbox.setChecked(preset.get("keep_models_loaded", True))
            self.lora_group.setChecked(preset.get("lora_enabled", False))
            self.lora_path.setText(preset.get("lora_path", ""))
            self.lora_scale.setValue(preset.get("lora_scale", 0.7))
            self.ti_token.setText(preset.get("ti_token", ""))
            self.update_status("Preset loaded.")

    def load_renderer_preset(self):
        preset = {
            "prompt": "A photorealistic render of a scene, high detail, studio lighting, PBR materials",
            "negative_prompt": "blurry, low detail, cartoon, sketch, painting, unrealistic, oversaturated",
            "base_model": "lykon/dreamshaper-xl-lightning",
            "adapter_models": [],
            "adapter_scale": 1.0,
            "scheduler": "DPM++ SDE Karras",
            "output_dir": hou.expandString("$HIP/render"),
            "width": 512,
            "height": 512,
            "strength": 0.4,
            "guidance_scale": 2.0,
            "num_steps": 4,
            "seed": -1,
            "keep_models_loaded": True,
            "lora_enabled": False,
            "lora_path": "",
            "lora_scale": 0.7,
            "ti_token": ""
        }
        self.prompt_input.setPlainText(preset["prompt"])
        self.negative_prompt.setPlainText(preset["negative_prompt"])
        index = self.model_combo.findText(preset["base_model"])
        if index != -1:
            self.model_combo.setCurrentIndex(index)
        for model, checkbox in self.adapter_checkboxes.items():
            checkbox.setChecked(model in preset["adapter_models"])
        self.adapter_scale.setValue(preset["adapter_scale"])
        index = self.scheduler_combo.findText(preset["scheduler"])
        if index != -1:
            self.scheduler_combo.setCurrentIndex(index)
        self.output_dir.setText(preset["output_dir"])
        self.width_spin.setValue(preset["width"])
        self.height_spin.setValue(preset["height"])
        self.strength.setValue(preset["strength"])
        self.guidance_scale.setValue(preset["guidance_scale"])
        self.num_steps.setValue(preset["num_steps"])
        self.seed_input.setValue(preset["seed"])
        self.keep_models_loaded_checkbox.setChecked(preset["keep_models_loaded"])
        self.lora_group.setChecked(preset["lora_enabled"])
        self.lora_path.setText(preset["lora_path"])
        self.lora_scale.setValue(preset["lora_scale"])
        self.ti_token.setText(preset["ti_token"])
        self.update_status("Renderer preset loaded.")

    def setup_performance_monitoring(self):
        if pynvml:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_handle = None
        self.perf_timer = QtCore.QTimer()
        self.perf_timer.timeout.connect(self.update_performance_metrics)
        self.perf_timer.start(1000)

    def update_performance_metrics(self):
        if self.gpu_handle:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                used_mb = mem_info.used // (1024 * 1024)
                total_mb = mem_info.total // (1024 * 1024)
                text = f"GPU Utilization: {util.gpu}% | Memory: {used_mb}MB / {total_mb}MB"
            except Exception as e:
                text = f"Error: {e}"
            self.performance_label.setText(text)
        else:
            self.performance_label.setText("GPU: N/A")

def optimize_pipeline(pipe):
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing(slice_size="auto")
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    if hasattr(pipe, 'vae') and callable(getattr(pipe.vae, 'enable_forward_chunking', None)):
        pipe.vae.enable_forward_chunking()
    return pipe

def onCreateInterface():
    return SDXLPanel()