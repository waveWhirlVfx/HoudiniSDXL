import hou
import torch
import threading
from PySide2 import QtWidgets, QtCore, QtGui
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverSDEScheduler

def optimize_pipeline(pipe):
    """Optimize the pipeline for memory efficiency and speed."""
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing(slice_size="auto")
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    if hasattr(pipe.vae, 'enable_forward_chunking'):
        pipe.vae.enable_forward_chunking()
    return pipe

class SimpleSDXLPanel(QtWidgets.QWidget):
    updateImageSignal = QtCore.Signal(QtGui.QImage)  # Signal to update image in UI
    modelLoadedSignal = QtCore.Signal()  # Signal to notify when model is loaded

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Image Generation")
        self.initUI()
        self.pipeline = None
        self.updateImageSignal.connect(self.onUpdateImage)  # Connect UI update signal
        self.modelLoadedSignal.connect(self.onModelLoaded)  # Connect model load signal

        # Start model loading in a separate thread
        threading.Thread(target=self.load_pipeline, daemon=True).start()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.prompt_label = QtWidgets.QLabel("Prompt:")
        self.prompt_input = QtWidgets.QTextEdit()
        self.prompt_input.setMaximumHeight(60)

        self.generate_btn = QtWidgets.QPushButton("Generate")
        self.generate_btn.clicked.connect(self.start_generation)
        # The generate button stays enabled at all times

        self.image_label = QtWidgets.QLabel("Generated image will appear here")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(256, 256)

        layout.addWidget(self.prompt_label)
        layout.addWidget(self.prompt_input)
        layout.addWidget(self.generate_btn)
        layout.addWidget(self.image_label)

    def load_pipeline(self):
        """Loads the model in a separate thread to prevent UI freeze."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "Lykon/dreamshaper-xl-v2-turbo"

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)

        # Set scheduler to DPM++ SDE Karras
        self.pipeline.scheduler = DPMSolverSDEScheduler(use_karras_sigmas=True)
        
        self.pipeline = optimize_pipeline(self.pipeline)

        # Emit signal to notify UI after loading
        self.modelLoadedSignal.emit()

    def onModelLoaded(self):
        """Notify that the model is loaded."""
        # Optionally update UI to indicate that model is ready
        print("Model loaded successfully.")

    def start_generation(self):
        if not self.pipeline:
            return
        # Generate button remains enabled; each click starts a new thread
        thread = threading.Thread(target=self.generate_image)
        thread.start()

    def generate_image(self):
        """Generates the image with the given prompt."""
        user_prompt = self.prompt_input.toPlainText()

        # Fixed cinematic quality prompts (Always included)
        fixed_prompt = "cinematic film still, (intense sunlight:1.4), realist detail, brooding mood, ue5, amazing quality, wallpaper, analog film grain"
        final_prompt = f"{user_prompt}, {fixed_prompt}" if user_prompt else fixed_prompt

        negative_prompt = "blurry, low detail, cartoon, sketch, painting, unrealistic, oversaturated"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(42)

        with torch.inference_mode():
            result = self.pipeline(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                height=1024,
                width=1024,
                guidance_scale=2,  # Fixed guidance scale
                num_inference_steps=8,
                generator=generator
            )

        # Convert PIL image to QImage
        pil_image = result.images[0]
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        data = pil_image.tobytes("raw", "RGB")
        width, height = pil_image.size
        qimage = QtGui.QImage(data, width, height, QtGui.QImage.Format_RGB888)

        # Emit the signal to update the UI with the image
        self.updateImageSignal.emit(qimage)

    def onUpdateImage(self, qimage):
        """Update the QLabel with the generated image, scaled to fit."""
        pixmap = QtGui.QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

def onCreateInterface():
    return SimpleSDXLPanel()
