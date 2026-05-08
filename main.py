import os

# Must be set before importing torch. Helps PyTorch avoid CUDA memory fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gc
import shutil
import time
from pathlib import Path
from typing import Any

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
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted, deformed"
CUDA_MEMORY_ERROR_MESSAGE = (
    "Not enough VRAM. Try LOW_VRAM_MODE=True, 512x512, 3 seconds, "
    "12-20 steps, close Chrome/VSCode."
)
DEVICE_MISMATCH_ERROR_MESSAGE = (
    "Prompt encoding device mismatch. Restart the app and try LOW_VRAM_MODE=True."
)

QUALITY_PRESETS = {
    "balanced": {
        "resolution": "768x512",
        "duration": 3,
        "steps": 24,
        "guidance_scale": 3.0,
        "low_vram": False,
    },
    "max_quality": {
        "resolution": "768x512",
        "duration": 5,
        "steps": 30,
        "guidance_scale": 3.0,
        "low_vram": False,
    },
    "low_vram": {
        "resolution": "512x512",
        "duration": 3,
        "steps": 16,
        "guidance_scale": 3.0,
        "low_vram": True,
    },
}

pipe = None
pipe_low_vram_mode: bool | None = None


def read_config_value(name: str, default: Any) -> Any:
    try:
        import config
    except Exception:
        return default
    return getattr(config, name, default)


def get_test_mode() -> bool:
    return bool(read_config_value("TEST_MODE", False))


def get_default_preset() -> str:
    preset = str(read_config_value("QUALITY_PRESET", "max_quality")).strip().lower()
    return preset if preset in QUALITY_PRESETS else "max_quality"


def get_default_settings() -> dict[str, Any]:
    preset_name = get_default_preset()
    preset = dict(QUALITY_PRESETS[preset_name])

    resolution = str(read_config_value("DEFAULT_RESOLUTION", preset["resolution"]))
    duration = int(read_config_value("DEFAULT_DURATION", preset["duration"]))
    steps = int(read_config_value("DEFAULT_STEPS", preset["steps"]))
    guidance_scale = float(read_config_value("DEFAULT_GUIDANCE_SCALE", preset["guidance_scale"]))

    low_vram_default = bool(preset["low_vram"])
    low_vram = bool(read_config_value("LOW_VRAM_MODE", low_vram_default))

    if resolution not in {"512x512", "768x512"}:
        resolution = preset["resolution"]
    if duration not in {3, 5}:
        duration = preset["duration"]
    steps = max(4, min(30, steps))
    guidance_scale = max(1.0, min(7.0, guidance_scale))

    return {
        "preset": preset_name,
        "resolution": resolution,
        "duration": duration,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "low_vram": low_vram,
    }


def preset_values(preset_name: str):
    preset = QUALITY_PRESETS.get(str(preset_name), QUALITY_PRESETS["balanced"])
    return (
        preset["resolution"],
        preset["duration"],
        preset["steps"],
        preset["guidance_scale"],
        preset["low_vram"],
    )


def is_cuda_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker.lower() in message
        for marker in (
            "out of memory",
            "cuda out of memory",
            "cudaerrormemoryallocation",
            "cublas_status_not_supported",
            "cublas_status_alloc_failed",
            "not enough memory",
        )
    )


def ensure_folders() -> None:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def validate_model_file() -> None:
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


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def unload_pipe() -> None:
    global pipe, pipe_low_vram_mode
    if pipe is not None:
        del pipe
    pipe = None
    pipe_low_vram_mode = None
    clear_cuda_cache()


def cuda_memory_snapshot() -> dict[str, str]:
    if not torch.cuda.is_available():
        return {
            "device": "CUDA unavailable",
            "total": "0.00 GB",
            "allocated": "0.00 GB",
            "reserved": "0.00 GB",
        }

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    gb = 1024 ** 3
    return {
        "device": props.name,
        "total": f"{props.total_memory / gb:.2f} GB",
        "allocated": f"{torch.cuda.memory_allocated(idx) / gb:.2f} GB",
        "reserved": f"{torch.cuda.memory_reserved(idx) / gb:.2f} GB",
    }


def format_status(settings: dict[str, Any], before: dict[str, str], after: dict[str, str] | None = None) -> str:
    lines = [
        f"Preset: {settings['preset']}",
        f"Resolution: {settings['resolution']}",
        f"Duration: {settings['duration']} sec",
        f"Frames: {settings['num_frames']}",
        f"Steps: {settings['steps']}",
        f"Guidance scale: {settings['guidance_scale']}",
        f"LOW_VRAM_MODE: {settings['low_vram']}",
        f"CUDA device: {before['device']}",
        f"VRAM total: {before['total']}",
        f"VRAM before: allocated={before['allocated']}, reserved={before['reserved']}",
    ]
    if after is not None:
        lines.append(f"VRAM after: allocated={after['allocated']}, reserved={after['reserved']}")
    return "\n".join(lines)


def load_tokenizer_and_text_encoder(low_vram_mode: bool):
    text_encoder_dtype = torch.float32 if low_vram_mode else torch.float16
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            str(TOKENIZER_DIR),
            local_files_only=True,
        )
        text_encoder = T5EncoderModel.from_pretrained(
            str(TEXT_ENCODER_DIR),
            torch_dtype=text_encoder_dtype,
            local_files_only=True,
        )
        if low_vram_mode:
            text_encoder.to("cpu")
            text_encoder.eval()
        return tokenizer, text_encoder
    except Exception as exc:
        raise gr.Error(
            "Failed to load local tokenizer/text_encoder.\n\n"
            "Check that these folders contain valid local Hugging Face component files:\n"
            f"- {TOKENIZER_DIR}\n"
            f"- {TEXT_ENCODER_DIR}\n\n"
            "The app is offline and will not download missing files."
        ) from exc


def configure_cuda_runtime() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def apply_memory_optimizations(model, low_vram_mode: bool):
    offload_enabled = False
    cpu_text_encoder = None

    # LOW_VRAM path keeps the text encoder on CPU and passes prompt embeddings manually.
    # This avoids wasting VRAM during prompt encoding.
    if low_vram_mode and hasattr(model, "text_encoder"):
        cpu_text_encoder = model.text_encoder
        model.text_encoder = None

    if low_vram_mode:
        if hasattr(model, "enable_sequential_cpu_offload"):
            model.enable_sequential_cpu_offload()
            offload_enabled = True
        elif hasattr(model, "enable_model_cpu_offload"):
            model.enable_model_cpu_offload()
            offload_enabled = True
        else:
            model.to(DEVICE)
    else:
        model.to(DEVICE)

    if cpu_text_encoder is not None:
        model.text_encoder = cpu_text_encoder
        model.text_encoder.to("cpu")
        model.text_encoder.eval()

    if hasattr(model, "vae") and hasattr(model.vae, "enable_tiling"):
        model.vae.enable_tiling()

    if hasattr(model, "vae") and hasattr(model.vae, "enable_slicing"):
        model.vae.enable_slicing()

    if hasattr(model, "enable_attention_slicing"):
        model.enable_attention_slicing("max")

    return offload_enabled


def load_pipe(low_vram_mode: bool):
    global pipe, pipe_low_vram_mode

    if pipe is not None and pipe_low_vram_mode == low_vram_mode:
        return pipe

    if pipe is not None and pipe_low_vram_mode != low_vram_mode:
        unload_pipe()

    validate_model_file()

    if not torch.cuda.is_available():
        raise gr.Error(
            "CUDA is not available. This project is configured for device=cuda.\n"
            "Check your NVIDIA driver and the CUDA build of PyTorch."
        )

    configure_cuda_runtime()
    tokenizer, text_encoder = load_tokenizer_and_text_encoder(low_vram_mode)

    try:
        model = LTXImageToVideoPipeline.from_single_file(
            str(MODEL_FILE),
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
    except Exception as exc:
        raise gr.Error(
            "Failed to load the local LTX-Video checkpoint.\n\n"
            f"Checkpoint:\n{MODEL_FILE}\n\n"
            "Check that the .safetensors file matches LTX-Video and is not corrupted.\n"
            "The app is offline and will not download replacement files."
        ) from exc

    try:
        apply_memory_optimizations(model, low_vram_mode)
    except torch.cuda.OutOfMemoryError as exc:
        unload_pipe()
        raise gr.Error(CUDA_MEMORY_ERROR_MESSAGE) from exc
    except Exception as exc:
        if is_cuda_memory_error(exc):
            unload_pipe()
            raise gr.Error(CUDA_MEMORY_ERROR_MESSAGE) from exc
        raise

    pipe = model
    pipe_low_vram_mode = low_vram_mode
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
    target = int(duration_seconds) * FPS
    return ((target - 2) // 8 + 1) * 8 + 1


def encode_prompts_on_cpu(model, prompt, negative_prompt, max_sequence_length=128):
    tokenizer = model.tokenizer
    text_encoder = model.text_encoder
    if tokenizer is None or text_encoder is None:
        raise gr.Error("LOW_VRAM_MODE requires a loaded tokenizer and text_encoder.")

    text_encoder.to("cpu")
    text_encoder.eval()

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="pt",
    )
    negative_text_inputs = tokenizer(
        negative_prompt,
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="pt",
    )

    with torch.inference_mode():
        prompt_embeds = text_encoder(text_inputs.input_ids)[0]
        negative_prompt_embeds = text_encoder(negative_text_inputs.input_ids)[0]

    if prompt_embeds.shape != negative_prompt_embeds.shape:
        raise gr.Error("Prompt and negative prompt embeddings have different shapes.")
    if text_inputs.attention_mask.shape != negative_text_inputs.attention_mask.shape:
        raise gr.Error("Prompt and negative prompt attention masks have different shapes.")

    return (
        prompt_embeds.to(dtype=torch.float16),
        negative_prompt_embeds.to(dtype=torch.float16),
        text_inputs.attention_mask,
        negative_text_inputs.attention_mask,
    )


def generate_video(image_path, prompt, quality_preset, low_vram_mode, duration, resolution, seed, steps, guidance_scale):
    ensure_folders()

    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt / scenario.")

    quality_preset = str(quality_preset or "balanced")
    low_vram_mode = bool(low_vram_mode)
    width, height = parse_resolution(resolution)
    duration = int(duration)
    steps = int(steps)
    guidance_scale = float(guidance_scale)
    num_frames = frame_count(duration)

    settings = {
        "preset": quality_preset,
        "resolution": resolution,
        "duration": duration,
        "num_frames": num_frames,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "low_vram": low_vram_mode,
    }
    before = cuda_memory_snapshot()

    saved_image = save_input_image(image_path)

    if get_test_mode():
        return None, format_status(settings, before) + "\n\nTest mode OK. Model generation skipped."

    clear_cuda_cache()
    before = cuda_memory_snapshot()

    try:
        model = load_pipe(low_vram_mode)
        image = load_image(str(saved_image))

        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        output_name = f"ltx_{time.strftime('%Y%m%d_%H%M%S')}_{width}x{height}_{duration}s.mp4"
        output_path = OUTPUTS_DIR / output_name

        with torch.inference_mode():
            if low_vram_mode:
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    prompt_attention_mask,
                    negative_prompt_attention_mask,
                ) = encode_prompts_on_cpu(model, prompt, NEGATIVE_PROMPT)

                execution_device = getattr(model, "_execution_device", torch.device(DEVICE))
                prompt_embeds = prompt_embeds.to(execution_device)
                negative_prompt_embeds = negative_prompt_embeds.to(execution_device)
                prompt_attention_mask = prompt_attention_mask.to(execution_device)
                negative_prompt_attention_mask = negative_prompt_attention_mask.to(execution_device)

                result = model(
                    image=image,
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    frame_rate=FPS,
                    num_inference_steps=steps,
                    generator=generator,
                    guidance_scale=guidance_scale,
                    decode_timestep=0.05,
                    decode_noise_scale=0.025,
                    max_sequence_length=128,
                )
            else:
                result = model(
                    image=image,
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    frame_rate=FPS,
                    num_inference_steps=steps,
                    generator=generator,
                    guidance_scale=guidance_scale,
                    decode_timestep=0.05,
                    decode_noise_scale=0.025,
                )

        export_to_video(result.frames[0], str(output_path), fps=FPS)
        after = cuda_memory_snapshot()
        clear_cuda_cache()

        return str(output_path), format_status(settings, before, after) + f"\n\nDone: {output_path}"

    except torch.cuda.OutOfMemoryError as exc:
        clear_cuda_cache()
        raise gr.Error(CUDA_MEMORY_ERROR_MESSAGE) from exc
    except Exception as exc:
        clear_cuda_cache()
        if "Expected all tensors to be on the same device" in str(exc):
            raise gr.Error(DEVICE_MISMATCH_ERROR_MESSAGE) from exc
        if is_cuda_memory_error(exc):
            raise gr.Error(CUDA_MEMORY_ERROR_MESSAGE) from exc
        raise


def build_ui():
    defaults = get_default_settings()

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
                quality_preset = gr.Dropdown(
                    choices=list(QUALITY_PRESETS.keys()),
                    value=defaults["preset"],
                    label="Quality preset",
                )
                low_vram_mode = gr.Checkbox(
                    label="LOW_VRAM_MODE / CPU offload",
                    value=defaults["low_vram"],
                )
                duration = gr.Radio(
                    choices=[3, 5],
                    value=defaults["duration"],
                    label="Duration, seconds",
                )
                resolution = gr.Radio(
                    choices=["512x512", "768x512"],
                    value=defaults["resolution"],
                    label="Resolution",
                )
                with gr.Accordion("Advanced", open=True):
                    seed = gr.Number(label="Seed", value=0, precision=0)
                    steps = gr.Slider(
                        label="Inference steps",
                        minimum=4,
                        maximum=30,
                        value=defaults["steps"],
                        step=1,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=1.0,
                        maximum=7.0,
                        value=defaults["guidance_scale"],
                        step=0.1,
                    )
                button = gr.Button("Generate", variant="primary")

            with gr.Column():
                video = gr.Video(label="Output video")
                status = gr.Textbox(label="Status / diagnostics", lines=14, interactive=False)

        quality_preset.change(
            fn=preset_values,
            inputs=[quality_preset],
            outputs=[resolution, duration, steps, guidance_scale, low_vram_mode],
        )

        button.click(
            fn=generate_video,
            inputs=[image, prompt, quality_preset, low_vram_mode, duration, resolution, seed, steps, guidance_scale],
            outputs=[video, status],
        )

    return demo


if __name__ == "__main__":
    ensure_folders()
    app = build_ui()
    app.launch(inbrowser=True)
