import gc
import os
import shutil
import time
from pathlib import Path

import gradio as gr
import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel, T5Tokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "ltx-video"
MODEL_FILE = MODEL_DIR / "ltx-video-2b-v0.9.safetensors"
TEXT_ENCODER_DIR = MODEL_DIR / "text_encoder"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "outputs"

FPS = 24
DEVICE = "cuda"
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

pipe = None


def get_test_mode():
    try:
        import config
    except Exception:
        return False

    return bool(getattr(config, "TEST_MODE", False))


def ensure_folders():
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def validate_model_file():
    if not MODEL_FILE.exists():
        raise gr.Error(
            "Missing LTX-Video checkpoint.\n\n"
            f"Expected file:\n{MODEL_FILE}\n\n"
            "Put your already downloaded checkpoint at:\n"
            "models/ltx-video/ltx-video-2b-v0.9.safetensors\n\n"
            "This app does not download model files automatically."
        )

    if not TEXT_ENCODER_DIR.exists():
        raise gr.Error(
            "Missing local text encoder.\n\n"
            f"Expected folder:\n{TEXT_ENCODER_DIR}\n\n"
            "Download/copy the LTX-Video text_encoder component into:\n"
            "models/ltx-video/text_encoder/\n\n"
            "Do not put it inside the .safetensors file. It must be a local folder."
        )

    if not TOKENIZER_DIR.exists():
        raise gr.Error(
            "Missing local tokenizer.\n\n"
            f"Expected folder:\n{TOKENIZER_DIR}\n\n"
            "Download/copy the LTX-Video tokenizer component into:\n"
            "models/ltx-video/tokenizer/\n\n"
            "This app does not download tokenizer files automatically."
        )


def load_pipe():
    global pipe
    if pipe is not None:
        return pipe

    validate_model_file()

    if not torch.cuda.is_available():
        raise gr.Error(
            "CUDA is not available. This project is configured for device=cuda.\n"
            "Check your NVIDIA driver and the CUDA build of PyTorch."
        )

    dtype = torch.float16

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        tokenizer = T5Tokenizer.from_pretrained(
            str(TOKENIZER_DIR),
            local_files_only=True,
        )
        text_encoder = T5EncoderModel.from_pretrained(
            str(TEXT_ENCODER_DIR),
            torch_dtype=dtype,
            local_files_only=True,
        )
    except Exception as exc:
        raise gr.Error(
            "Failed to load local tokenizer/text_encoder.\n\n"
            "Check that these folders contain valid local Hugging Face component files:\n"
            f"- {TOKENIZER_DIR}\n"
            f"- {TEXT_ENCODER_DIR}\n\n"
            "The app is offline and will not download missing files."
        ) from exc

    try:
        pipe = LTXImageToVideoPipeline.from_single_file(
            str(MODEL_FILE),
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=dtype,
            local_files_only=True,
        )
    except Exception as exc:
        raise gr.Error(
            "Failed to load the local LTX-Video checkpoint.\n\n"
            f"Checkpoint:\n{MODEL_FILE}\n\n"
            "Check that the .safetensors file matches LTX-Video and is not corrupted.\n"
            "The app is offline and will not download replacement files."
        ) from exc

    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(DEVICE)

    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    return pipe


def save_input_image(image_path):
    if image_path is None:
        raise gr.Error("Upload an input image.")

    source = Path(image_path)
    if not source.exists():
        raise gr.Error("Input image file was not found.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = INPUTS_DIR / f"input_{timestamp}{source.suffix.lower() or '.png'}"
    shutil.copy2(source, target)
    return target


def parse_resolution(resolution):
    width, height = resolution.lower().split("x")
    return int(width), int(height)


def frame_count(duration_seconds):
    # LTX works best with frame counts in the form 8n + 1.
    target = duration_seconds * FPS
    return ((target - 2) // 8 + 1) * 8 + 1


def generate_video(image_path, prompt, duration, resolution, seed, steps):
    ensure_folders()

    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt / scenario.")

    saved_image = save_input_image(image_path)
    width, height = parse_resolution(resolution)
    num_frames = frame_count(int(duration))

    if get_test_mode():
        return None, "Test mode OK. Model generation skipped."

    model = load_pipe()
    image = load_image(str(saved_image))

    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    output_name = f"ltx_{time.strftime('%Y%m%d_%H%M%S')}_{width}x{height}_{duration}s.mp4"
    output_path = OUTPUTS_DIR / output_name

    try:
        with torch.inference_mode():
            result = model(
                image=image,
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=FPS,
                num_inference_steps=int(steps),
                generator=generator,
                guidance_scale=3.0,
                decode_timestep=0.05,
                decode_noise_scale=0.025,
            )

        export_to_video(result.frames[0], str(output_path), fps=FPS)

    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise gr.Error(
            "Not enough VRAM for generation.\n"
            "Try 512x512, 3 seconds, or fewer inference steps."
        ) from exc

    return str(output_path), f"Done: {output_path}"


def build_ui():
    with gr.Blocks(title="Local LTX-Video") as demo:
        gr.Markdown("# Local LTX-Video")
        gr.Markdown("Image-to-video: upload an image, write a prompt, then click Generate.")

        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Input image", type="filepath")
                prompt = gr.Textbox(
                    label="Prompt / scenario",
                    lines=5,
                    placeholder="Describe the motion, camera movement, scene details and lighting...",
                )
                duration = gr.Radio(
                    choices=[3, 5],
                    value=3,
                    label="Duration, seconds",
                )
                resolution = gr.Radio(
                    choices=["512x512", "768x512"],
                    value="512x512",
                    label="Resolution",
                )
                with gr.Accordion("Advanced", open=False):
                    seed = gr.Number(label="Seed", value=0, precision=0)
                    steps = gr.Slider(
                        label="Inference steps",
                        minimum=4,
                        maximum=30,
                        value=4,
                        step=1,
                    )
                button = gr.Button("Generate", variant="primary")

            with gr.Column():
                video = gr.Video(label="Output video")
                status = gr.Textbox(label="Status", interactive=False)

        button.click(
            fn=generate_video,
            inputs=[image, prompt, duration, resolution, seed, steps],
            outputs=[video, status],
        )

    return demo


if __name__ == "__main__":
    ensure_folders()
    app = build_ui()
    app.launch(inbrowser=True)
