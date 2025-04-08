import hou
import torch
import torch.nn as nn
import threading
import logging
import os
import datetime
import time
import json
import shutil
from PIL import Image, ImageFilter, ImageOps, ImageChops
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLAdapterPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    T2IAdapter
)
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from huggingface_hub import HfApi, snapshot_download, login

try:
    import pynvml
    pynvml.nvmlInit()
except Exception as e:
    pynvml = None

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pruna(pipe):
    logger.debug("Applying DeepCache pruna optimization to the pipeline.")
    return pipe

def optimize_pipeline(pipe):
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing(slice_size="auto")
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    if hasattr(pipe, 'vae') and callable(getattr(pipe.vae, 'enable_forward_chunking', None)):
        pipe.vae.enable_forward_chunking()
    pipe = pruna(pipe)
    return pipe

def load_lora_weights(pipe, lora_path, lora_scale=1.0, hf_token=None):
    try:
        if os.path.exists(lora_path):
            logger.debug(f"Loading LoRA weights from local file: {lora_path}")
            pipe.load_lora_weights(lora_path)
        elif "/" in lora_path:
            logger.debug(f"Downloading LoRA weights from repo: {lora_path}")
            local_dir = snapshot_download(repo_id=lora_path, allow_patterns="*.safetensors", token=hf_token)
            files = [f for f in os.listdir(local_dir) if f.endswith(".safetensors")]
            if not files:
                raise ValueError(f"No safetensors file found in snapshot download for repo {lora_path}")
            downloaded_file = os.path.join(local_dir, files[0])
            pipe.load_lora_weights(downloaded_file)
        else:
            raise ValueError("LoRA path provided is neither a local file nor a valid repo id.")
        return pipe, lora_scale
    except Exception as e:
        logger.error(f"Error loading LoRA weights: {str(e)}")
        raise

from PySide2 import QtWidgets, QtCore, QtGui

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
                local_path = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                                          f"models--{model.modelId.replace('/', '--')}")
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
                snapshot_download(repo_id=model_id, token=self.huggingface_token)
                self.status_label.setText(f"Downloaded {model_id}")
                self.refresh_model_list()
            except Exception as e:
                self.status_label.setText(f"Download error: {str(e)}")

    def delete_model(self):
        selected = self.model_list.currentItem()
        if selected:
            model_id = selected.text()
            local_path = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                                      f"models--{model_id.replace('/', '--')}")
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

class SDXLPanel(QtWidgets.QWidget):
    progress_signal = QtCore.Signal(int)
    status_signal = QtCore.Signal(str)
    enable_render_signal = QtCore.Signal(bool)
    enable_cancel_signal = QtCore.Signal(bool)
    input_preview_signal = QtCore.Signal(QtGui.QPixmap)
    output_preview_signal = QtCore.Signal(QtGui.QPixmap)
    history_signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.cancel_flag = False
        self.adapter_checkboxes = {}
        self.huggingface_token = ""
        self.cached_pipe = None
        self.cached_config = None
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
        self.input_preview_signal.connect(self.input_preview.setPixmap)
        self.output_preview_signal.connect(self.output_preview.setPixmap)
        self.history_signal.connect(lambda text: self.history_list.addItem(text))

    def initUI(self):
        mainWidget = QtWidgets.QWidget()
        mainLayout = QtWidgets.QVBoxLayout(mainWidget)
        tab_widget = QtWidgets.QTabWidget()
        generation_tab = QtWidgets.QWidget()
        gen_layout = QtWidgets.QVBoxLayout(generation_tab)
        top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        prompt_widget = QtWidgets.QWidget()
        prompt_layout = QtWidgets.QVBoxLayout(prompt_widget)
        prompt_layout.setContentsMargins(5, 5, 5, 5)
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
        preview_layout.setContentsMargins(5, 5, 5, 5)
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
        params_layout.setContentsMargins(5, 5, 5, 5)
        row = 0
        params_layout.addWidget(QtWidgets.QLabel("Base Model:"), row, 0)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems([
            "stabilityai/stable-diffusion-xl-base-1.0",
            "dreamlike-art/dreamlike-photoreal-2.0",
            "lykon/dreamshaper-xl-lightning",
            "Lykon/dreamshaper-xl-v2-turbo",
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1"
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
        self.maps_checkbox = QtWidgets.QCheckBox("Generate Maps (Bump, Specular, Roughness, Displacement)")
        self.maps_checkbox.setChecked(False)
        gen_layout.addWidget(self.maps_checkbox)
        control_group = QtWidgets.QGroupBox("Control Models (Select only one)")
        control_group.setCheckable(True)
        control_group.setChecked(False)
        control_group.toggled.connect(lambda checked: [child.setVisible(checked) for child in control_group.findChildren(QtWidgets.QWidget)])
        control_layout = QtWidgets.QVBoxLayout(control_group)
        control_models = [
            "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
            "TencentARC/t2i-adapter-canny-sdxl-1.0",
            "TencentARC/t2i-adapter-sketch-sdxl-1.0",
            "TencentARC/t2i-adapter-seg-sdxl-1.0",
            "TencentARC/t2i-adapter-lineart-sdxl-1.0"
        ]
        control_grid = QtWidgets.QGridLayout()
        row = 0
        col = 0
        self.control_group = QtWidgets.QButtonGroup()
        self.control_group.setExclusive(True)
        for model in control_models:
            short_name = model.split('/')[-1].replace('t2i-adapter-', '').replace('-sdxl-1.0', '')
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
        self.lora_group.toggled.connect(lambda checked: [child.setVisible(checked) for child in self.lora_group.findChildren(QtWidgets.QWidget) if child != self.lora_group])
        lora_layout = QtWidgets.QVBoxLayout(self.lora_group)
        lora_file_layout = QtWidgets.QHBoxLayout()
        lora_file_layout.addWidget(QtWidgets.QLabel("LoRA Model:"))
        self.lora_path = QtWidgets.QLineEdit()
        self.lora_path.setPlaceholderText("Enter repo id (e.g., PvDeep/Add-Detail-XL) or browse a file")
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
        gen_layout.addWidget(self.lora_group)
        self.maps_group = QtWidgets.QGroupBox("Maps Preview")
        self.maps_group.setVisible(False)
        self.maps_checkbox.toggled.connect(lambda checked: self.maps_group.setVisible(checked))
        maps_layout = QtWidgets.QHBoxLayout(self.maps_group)
        bump_layout = QtWidgets.QVBoxLayout()
        bump_layout.addWidget(QtWidgets.QLabel("Bump Map"))
        self.bump_preview = ImagePreviewLabel()
        bump_layout.addWidget(self.bump_preview)
        specular_layout = QtWidgets.QVBoxLayout()
        specular_layout.addWidget(QtWidgets.QLabel("Specular Map"))
        self.specular_preview = ImagePreviewLabel()
        specular_layout.addWidget(self.specular_preview)
        roughness_layout = QtWidgets.QVBoxLayout()
        roughness_layout.addWidget(QtWidgets.QLabel("Roughness Map"))
        self.roughness_preview = ImagePreviewLabel()
        roughness_layout.addWidget(self.roughness_preview)
        displacement_layout = QtWidgets.QVBoxLayout()
        displacement_layout.addWidget(QtWidgets.QLabel("Displacement Map"))
        self.displacement_preview = ImagePreviewLabel()
        displacement_layout.addWidget(self.displacement_preview)
        maps_layout.addLayout(bump_layout)
        maps_layout.addLayout(specular_layout)
        maps_layout.addLayout(roughness_layout)
        maps_layout.addLayout(displacement_layout)
        gen_layout.addWidget(self.maps_group)
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
        self.log_widget = QtWidgets.QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(150)
        self.log_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        gen_layout.addWidget(QtWidgets.QLabel("Log:"))
        gen_layout.addWidget(self.log_widget)
        tab_widget.addTab(generation_tab, "Generate")
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
        tab_widget.addTab(presets_tab, "Presets & History")
        mainLayout.addWidget(tab_widget)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(mainWidget)
        scroll.setWidgetResizable(True)
        outerLayout = QtWidgets.QVBoxLayout(self)
        outerLayout.addWidget(scroll)

    def get_pipeline_key(self):
        key = self.model_combo.currentText()
        selected_adapters = self.get_selected_adapter_models()
        if selected_adapters:
            key += "_adapter_" + "_".join(sorted(selected_adapters))
        else:
            key += "_base"
        if self.lora_group.isChecked() and self.lora_path.text():
            key += f"_lora_{self.lora_path.text()}"
        return key

    def create_bump_map(self, image):
        gray = image.convert("L")
        bump = gray.filter(ImageFilter.EMBOSS)
        bump = ImageOps.autocontrast(bump)
        return bump

    def create_specular_map(self, image):
        gray = image.convert("L")
        spec = gray.point(lambda x: 255 if x > 180 else int(x * 0.5))
        spec = ImageOps.autocontrast(spec)
        return spec

    def create_roughness_map(self, image):
        spec = self.create_specular_map(image)
        roughness = ImageOps.invert(spec)
        roughness = ImageOps.autocontrast(roughness)
        return roughness

    def create_displacement_map(self, image):
        gray = image.convert("L")
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=5))
        displacement = ImageChops.difference(gray, blurred)
        displacement = ImageOps.autocontrast(displacement)
        return displacement

    def create_principled_shader(self, base_path, bump_path, specular_path, roughness_path, displacement_path):
        matnet = hou.node("/mat")
        if matnet is None:
            matnet = hou.node("/shop").createNode("matnet")
        shader = matnet.createNode("principledshader::2.0", "generated_principled")
        shader.parm("basecolor_useTexture").set(1)
        shader.parm("basecolor_texture").set(base_path)
        shader.parm("baseBumpAndNormal_enable").set(1)
        shader.parm("baseNormal_texture").set(bump_path)
        shader.parm("reflect_useTexture").set(1)
        shader.parm("reflect_texture").set(specular_path)
        shader.parm("rough_useTexture").set(1)
        shader.parm("rough_texture").set(roughness_path)
        shader.parm("dispTex_enable").set(1)
        shader.parm("dispTex_texture").set(displacement_path)
        shader.moveToGoodPosition()
        self.status_signal.emit("Created Principled Shader 'generated_principled' in /mat")
        return shader

    def generate_caption(self):
        output_dir = self.output_dir.text()
        image = self.capture_viewport(output_dir)
        if image is None:
            self.status_signal.emit("Failed to capture viewport for captioning")
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
        self.status_signal.emit("Caption generated and added to prompt")

    def progress_callback(self, step, timestep, latents):
        total_steps = self.num_steps.value()
        progress = int((step / total_steps) * 100)
        self.progress_signal.emit(progress)
        QtWidgets.QApplication.processEvents()

    def generate_image(self):
        start_time = time.time()
        pipe = None
        lora_scale = self.lora_scale.value()  # retrieve lora scale value
        torch.cuda.empty_cache()
        output_dir = self.output_dir.text()
        os.makedirs(output_dir, exist_ok=True)
        selected_models = self.get_selected_adapter_models()
        lora_enabled = self.lora_group.isChecked() and self.lora_path.text()
        cond_img = None
        if selected_models:
            self.status_signal.emit("Capturing viewport...")
            cond_img = self.capture_viewport(output_dir)
            if cond_img is None:
                self.status_signal.emit("Failed to capture viewport!")
                return
            pixmap = self.pil_image_to_pixmap(cond_img)
            self.input_preview_signal.emit(pixmap)
        try:
            device = "cpu" if self.use_cpu_only else "cuda"
            hf_token = self.huggingface_token if self.huggingface_token else None
            torch_dtype = torch.float16
            key = self.get_pipeline_key()
            if self.keep_models_loaded_checkbox.isChecked() and self.cached_pipe is not None and self.cached_config == key:
                pipe = self.cached_pipe
                self.status_signal.emit("Using cached pipeline.")
            else:
                if selected_models:
                    adapter = T2IAdapter.from_pretrained(
                        selected_models[0],
                        torch_dtype=torch_dtype,
                        variant="fp16",
                        use_auth_token=hf_token
                    ).to(device)
                    for param in adapter.parameters():
                        param.requires_grad = False
                    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
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
            if lora_enabled:
                self.status_signal.emit("Loading LoRA weights...")
                pipe, lora_scale = load_lora_weights(pipe, self.lora_path.text(), lora_scale, hf_token)
            scheduler_name = self.scheduler_combo.currentText()
            scheduler_config = self.get_scheduler_config(scheduler_name)
            pipe.scheduler = scheduler_config.from_config(pipe.scheduler.config)
            pipe = optimize_pipeline(pipe)
            if self.keep_models_loaded_checkbox.isChecked():
                self.cached_pipe = pipe
                self.cached_config = key
            else:
                self.cached_pipe = None
                self.cached_config = None
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
            if not selected_models:
                generation_args["strength"] = self.strength.value() if cond_img else 1.0
            if self.seed_input.value() != -1:
                generation_args["generator"] = torch.Generator(device).manual_seed(self.seed_input.value())
            if selected_models:
                generation_args["adapter_conditioning_scale"] = self.adapter_scale.value()
                generation_args["image"] = cond_img
            self.status_signal.emit("Generating image...")
            self.progress_bar.setVisible(True)
            with torch.inference_mode():
                result = pipe(**generation_args)
            elapsed_time = time.time() - start_time
            if not self.cancel_flag and result.images:
                final_image = result.images[0]
                pixmap = self.pil_image_to_pixmap(final_image)
                self.output_preview_signal.emit(pixmap)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base_path = os.path.join(output_dir, f"generated_{timestamp}.png")
                final_image.save(base_path)
                if self.maps_checkbox.isChecked():
                    bump_map = self.create_bump_map(final_image)
                    specular_map = self.create_specular_map(final_image)
                    roughness_map = self.create_roughness_map(final_image)
                    displacement_map = self.create_displacement_map(final_image)
                    bump_path = os.path.join(output_dir, f"generated_bump_{timestamp}.png")
                    specular_path = os.path.join(output_dir, f"generated_specular_{timestamp}.png")
                    roughness_path = os.path.join(output_dir, f"generated_roughness_{timestamp}.png")
                    displacement_path = os.path.join(output_dir, f"generated_displacement_{timestamp}.png")
                    bump_map.save(bump_path)
                    specular_map.save(specular_path)
                    roughness_map.save(roughness_path)
                    displacement_map.save(displacement_path)
                    self.bump_preview.setPixmap(self.pil_image_to_pixmap(bump_map.convert("RGB")))
                    self.specular_preview.setPixmap(self.pil_image_to_pixmap(specular_map.convert("RGB")))
                    self.roughness_preview.setPixmap(self.pil_image_to_pixmap(roughness_map.convert("RGB")))
                    self.displacement_preview.setPixmap(self.pil_image_to_pixmap(displacement_map.convert("RGB")))
                    self.status_signal.emit(
                        f"Base image saved: {base_path} | "
                        f"Bump: {bump_path}, Specular: {specular_path}, "
                        f"Roughness: {roughness_path}, Displacement: {displacement_path} | "
                        f"Took {elapsed_time:.2f} seconds"
                    )
                    self.create_principled_shader(base_path, bump_path, specular_path, roughness_path, displacement_path)
                else:
                    self.status_signal.emit(f"Base image saved: {base_path} | Took {elapsed_time:.2f} seconds")
                self.history.append(f"{timestamp} - {base_path} - Seed: {self.seed_input.value()}")
                self.history_signal.emit(f"{timestamp} - {base_path} - Seed: {self.seed_input.value()}")
        except torch.cuda.OutOfMemoryError as e:
            self.status_signal.emit("GPU memory exceeded. Try lower resolution, fewer steps, or CPU-only mode.")
        except Exception as e:
            self.status_signal.emit(f"Error: {str(e)}")
        finally:
            self.enable_render_signal.emit(True)
            self.enable_cancel_signal.emit(False)
            self.progress_bar.setVisible(False)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, text):
        self.status_label.setText(text)
        self.log_widget.appendPlainText(text)
        logger.info(f"Status updated: {text}")

    def set_render_enabled(self, enabled):
        self.render_btn.setEnabled(enabled)

    def set_cancel_enabled(self, enabled):
        self.cancel_btn.setEnabled(enabled)

    def pil_image_to_pixmap(self, pil_image):
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        data = pil_image.tobytes("raw", "RGB")
        qimage = QtGui.QImage(data, pil_image.size[0], pil_image.size[1], QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimage)

    def start_generation(self):
        torch.cuda.empty_cache()
        self.cancel_flag = False
        self.render_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_signal.emit(0)
        self.status_signal.emit("Initializing...")
        thread = threading.Thread(target=self.threaded_generation)
        thread.start()

    def threaded_generation(self):
        self.generate_image()

    def cancel_generation(self):
        self.cancel_flag = True
        self.status_signal.emit("Canceling...")

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
                self.status_signal.emit("No scene viewer found!")
                return None
            viewport = scene_viewer.curViewport()
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(output_dir, f"viewport_{timestamp}.jpg")
            try:
                self.status_signal.emit("Capturing viewport using viewwrite...")
                camera_path = f"{cur_desktop.name()}.{scene_viewer.name()}.world.{viewport.name()}"
                frame = hou.frame()
                hou.hscript(f'viewwrite -r 512 512 -f {frame} {frame} {camera_path} "{filename}"')
            except Exception as e:
                self.status_signal.emit("Trying alternate capture method...")
                viewport.saveFrame(filename)
            if os.path.exists(filename):
                self.status_signal.emit("Loading captured image...")
                image = Image.open(filename)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                if image.size[0] < 512 or image.size[1] < 512:
                    image = image.resize((512, 512), Image.Resampling.LANCZOS)
                return image
            else:
                self.status_signal.emit("Failed to save viewport image!")
                return None
        except Exception as e:
            logger.error(f"Viewport capture error: {str(e)}")
            self.status_signal.emit(f"Viewport capture error: {str(e)}")
            return None

    def get_selected_adapter_models(self):
        selected = [model for model, checkbox in self.adapter_checkboxes.items() if checkbox.isChecked()]
        if len(selected) > 1:
            raise ValueError("Only one control model can be selected at a time.")
        return selected

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
            "lora_scale": self.lora_scale.value()
        }
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset", "Preset Name:")
        if ok and name:
            presets_dir = os.path.join(os.getcwd(), "presets")
            if not os.path.exists(presets_dir):
                os.makedirs(presets_dir)
            file_path = os.path.join(presets_dir, f"{name}.json")
            with open(file_path, "w") as f:
                json.dump(preset, f)
            self.status_signal.emit(f"Preset '{name}' saved.")

    def load_preset(self):
        presets_dir = os.path.join(os.getcwd(), "presets")
        if not os.path.exists(presets_dir):
            self.status_signal.emit("No presets found.")
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
                self.status_signal.emit("Warning: Preset contains multiple adapters; only the first will be used.")
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
            self.status_signal.emit("Preset loaded.")

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
            "lora_scale": 0.7
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
        self.status_signal.emit("Renderer preset loaded.")

    def open_settings(self):
        token, ok = QtWidgets.QInputDialog.getText(self, "Settings", "Hugging Face Token:")
        if ok and token:
            self.huggingface_token = token
            login(token=self.huggingface_token)
            self.status_signal.emit("Settings updated and logged in to Hugging Face.")

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
        self.status_signal.emit(f"CPU-only mode: {'Enabled' if self.use_cpu_only else 'Disabled'}")
        logger.info(f"CPU-only mode toggled to: {self.use_cpu_only}")

    def setup_performance_monitoring(self):
        if pynvml:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
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
            self.status_label.setText(text)
        else:
            self.status_label.setText("GPU: N/A")

def onCreateInterface():
    return SDXLPanel()
