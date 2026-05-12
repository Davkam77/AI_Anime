import gc
import os
import shutil
import time
from pathlib import Path
from typing import Any

# Must be set before importing torch.
# Some Windows/PyTorch builds ignore expandable_segments; warning is safe.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Allow tiny Diffusers config downloads by default.
# The heavy model files are still loaded from local MODEL_FILE.
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

import gradio as gr
import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel, T5Tokenizer


BASE_DIR = Path(__file__).resolve().parent

# Preferred model search order.
# Put the new lightweight model here:
#   models/ltx-video-fp8/ltxv-2b-0.9.8-distilled-fp8.safetensors
# or here:
#   models/ltx-video/ltxv-2b-0.9.8-distilled-fp8.safetensors
DEFAULT_MODEL_CANDIDATES = [
    BASE_DIR / "models" / "ltx-video-fp8" / "ltxv-2b-0.9.8-distilled-fp8.safetensors",
    BASE_DIR / "models" / "ltx-video" / "ltxv-2b-0.9.8-distilled-fp8.safetensors",
    BASE_DIR / "models" / "ltx-video" / "ltxv-2b-0.9.8-distilled.safetensors",
    BASE_DIR / "models" / "ltx-video" / "ltx-video-2b-v0.9.safetensors",
]

INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "outputs"

FPS = 24
DEVICE = "cuda"
FORCE_LOW_VRAM_GB = 10.5

NEGATIVE_PROMPT = (
    "blurry, distorted, morphing, deformed character, unstable shape, flicker, "
    "jittery motion, camera shake, low quality, melted details, text, watermark, logo"
)

CUDA_MEMORY_ERROR_MESSAGE = (
    "Not enough GPU memory for this mode.\n\n"
    "Use LOW_VRAM_MODE=True, 512x512, 3 seconds, 8-16 steps, guidance 3.0.\n"
    "Close Chrome/VSCode/Discord and try again."
)

DEVICE_MISMATCH_ERROR_MESSAGE = (
    "Prompt encoding device mismatch.\n\n"
    "This usually means text_encoder is on CPU but the pipeline tried to encode prompt on CUDA.\n"
    "Use LOW_VRAM_MODE=True. This build pre-encodes prompts on CPU in low-vram mode."
)


QUALITY_PRESETS = {
    "low_vram": {
        "resolution": "512x512",
        "duration": 3,
        "steps": 12,
        "guidance_scale": 3.0,
        "low_vram": True,
    },
    "balanced": {
        "resolution": "512x512",
        "duration": 3,
        "steps": 16,
        "guidance_scale": 3.0,
        "low_vram": True,
    },
    "max_quality": {
        "resolution": "768x512",
        "duration": 3,
        "steps": 24,
        "guidance_scale": 3.0,
        "low_vram": True,
    },
}

pipe = None
pipe_key: tuple[str, bool] | None = None


def read_config_value(name: str, default: Any) -> Any:
    try:
        import config
    except Exception:
        return default
    return getattr(config, name, default)


def get_test_mode() -> bool:
    return bool(read_config_value("TEST_MODE", False))


def get_configured_model_file() -> Path | None:
    raw_model_file = read_config_value("MODEL_FILE", None)
    if raw_model_file:
        path = Path(str(raw_model_file))
        if not path.is_absolute():
            path = BASE_DIR / path
        return path

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    return None


def get_model_dir(model_file: Path | None = None) -> Path:
    if model_file is None:
        model_file = get_configured_model_file()

    if model_file is not None:
        return model_file.parent

    return BASE_DIR / "models" / "ltx-video"


def get_component_dir(name: str, model_file: Path | None = None) -> Path:
    configured = read_config_value(f"{name.upper()}_DIR", None)
    if configured:
        path = Path(str(configured))
        if not path.is_absolute():
            path = BASE_DIR / path
        return path

    model_dir = get_model_dir(model_file)
    candidate = model_dir / name
    if candidate.exists():
        return candidate

    fallback = BASE_DIR / "models" / "ltx-video" / name
    if fallback.exists():
        return fallback

    return candidate


def get_default_preset() -> str:
    preset = str(read_config_value("QUALITY_PRESET", "low_vram")).strip().lower()
    return preset if preset in QUALITY_PRESETS else "low_vram"


def should_force_low_vram() -> bool:
    if not torch.cuda.is_available():
        return True

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_gb = props.total_memory / (1024 ** 3)
    return total_gb <= FORCE_LOW_VRAM_GB


def get_default_settings() -> dict[str, Any]:
    preset_name = get_default_preset()
    preset = dict(QUALITY_PRESETS[preset_name])

    resolution = str(read_config_value("DEFAULT_RESOLUTION", preset["resolution"]))
    duration = int(read_config_value("DEFAULT_DURATION", preset["duration"]))
    steps = int(read_config_value("DEFAULT_STEPS", preset["steps"]))
    guidance_scale = float(read_config_value("DEFAULT_GUIDANCE_SCALE", preset["guidance_scale"]))

    low_vram_default = bool(preset["low_vram"])
    low_vram = bool(read_config_value("LOW_VRAM_MODE", low_vram_default))
    if should_force_low_vram():
        low_vram = True

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
    preset = QUALITY_PRESETS.get(str(preset_name), QUALITY_PRESETS["low_vram"])
    low_vram = bool(preset["low_vram"])
    if should_force_low_vram():
        low_vram = True

    return (
        preset["resolution"],
        preset["duration"],
        preset["steps"],
        preset["guidance_scale"],
        low_vram,
    )


def is_cuda_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
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
    (BASE_DIR / "models").mkdir(parents=True, exist_ok=True)


def validate_model_files(model_file: Path, tokenizer_dir: Path, text_encoder_dir: Path) -> None:
    if not model_file.exists():
        candidates = "\n".join(f"- {p}" for p in DEFAULT_MODEL_CANDIDATES)
        raise gr.Error(
            "Missing LTX-Video checkpoint.\n\n"
            "Expected one of these files:\n"
            f"{candidates}\n\n"
            "Recommended for RTX 3060 Ti 8GB:\n"
            "models/ltx-video-fp8/ltxv-2b-0.9.8-distilled-fp8.safetensors"
        )

    if not tokenizer_dir.exists():
        raise gr.Error(
            "Missing local tokenizer folder.\n\n"
            f"Expected folder:\n{tokenizer_dir}\n\n"
            "Copy/download tokenizer into this folder."
        )

    if not text_encoder_dir.exists():
        raise gr.Error(
            "Missing local text_encoder folder.\n\n"
            f"Expected folder:\n{text_encoder_dir}\n\n"
            "Copy/download text_encoder into this folder."
        )


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def unload_pipe() -> None:
    global pipe, pipe_key
    if pipe is not None:
        del pipe
    pipe = None
    pipe_key = None
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
        f"Model: {settings['model_file']}",
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


def configure_cuda_runtime() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_tokenizer_and_text_encoder(tokenizer_dir: Path, text_encoder_dir: Path, low_vram_mode: bool):
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            str(tokenizer_dir),
            local_files_only=True,
        )

        text_encoder_dtype = torch.float32 if low_vram_mode else torch.float16
        text_encoder = T5EncoderModel.from_pretrained(
            str(text_encoder_dir),
            torch_dtype=text_encoder_dtype,
            local_files_only=True,
        )

        if low_vram_mode:
            text_encoder.to("cpu")
        else:
            text_encoder.to(DEVICE)

        text_encoder.eval()
        return tokenizer, text_encoder

    except Exception as exc:
        raise gr.Error(
            "Failed to load local tokenizer/text_encoder.\n\n"
            f"Tokenizer:\n{tokenizer_dir}\n\n"
            f"Text encoder:\n{text_encoder_dir}"
        ) from exc


def apply_memory_optimizations(model, low_vram_mode: bool) -> str:
    if hasattr(model, "remove_all_hooks"):
        try:
            model.remove_all_hooks()
        except Exception:
            pass

    if low_vram_mode:
        text_encoder = getattr(model, "text_encoder", None)

        if text_encoder is not None:
            text_encoder.to("cpu")
            text_encoder.eval()

        # Prevent Accelerate hooks from moving T5 to CUDA.
        model.text_encoder = None

        if hasattr(model, "enable_sequential_cpu_offload"):
            model.enable_sequential_cpu_offload()
            offload_mode = "sequential_cpu_offload"
        elif hasattr(model, "enable_model_cpu_offload"):
            model.enable_model_cpu_offload()
            offload_mode = "model_cpu_offload"
        else:
            # Last fallback. This can OOM.
            model.to(DEVICE)
            offload_mode = "cuda_full_fallback"

        model.text_encoder = text_encoder
        if model.text_encoder is not None:
            model.text_encoder.to("cpu")
            model.text_encoder.eval()

        if hasattr(model, "vae") and model.vae is not None:
            if hasattr(model.vae, "enable_tiling"):
                try:
                    model.vae.enable_tiling()
                except Exception:
                    pass
            if hasattr(model.vae, "enable_slicing"):
                try:
                    model.vae.enable_slicing()
                except Exception:
                    pass

        print("LOW_VRAM_MODE:", True)
        print("OFFLOAD_MODE:", offload_mode)
        print("TEXT_ENCODER:", "cpu")
        return offload_mode

    model.to(DEVICE)

    if hasattr(model, "vae") and model.vae is not None:
        if hasattr(model.vae, "disable_slicing"):
            try:
                model.vae.disable_slicing()
            except Exception:
                pass
        if hasattr(model.vae, "disable_tiling"):
            try:
                model.vae.disable_tiling()
            except Exception:
                pass

    print("LOW_VRAM_MODE:", False)
    print("OFFLOAD_MODE:", "full_gpu")
    return "full_gpu"


def load_pipe(low_vram_mode: bool):
    global pipe, pipe_key

    model_file = get_configured_model_file()
    if model_file is None:
        model_file = DEFAULT_MODEL_CANDIDATES[0]

    tokenizer_dir = get_component_dir("tokenizer", model_file)
    text_encoder_dir = get_component_dir("text_encoder", model_file)

    current_key = (str(model_file), bool(low_vram_mode))
    if pipe is not None and pipe_key == current_key:
        return pipe

    if pipe is not None and pipe_key != current_key:
        unload_pipe()

    validate_model_files(model_file, tokenizer_dir, text_encoder_dir)

    if not torch.cuda.is_available():
        raise gr.Error(
            "CUDA is not available. Check NVIDIA driver and CUDA PyTorch build."
        )

    configure_cuda_runtime()
    tokenizer, text_encoder = load_tokenizer_and_text_encoder(
        tokenizer_dir=tokenizer_dir,
        text_encoder_dir=text_encoder_dir,
        low_vram_mode=low_vram_mode,
    )

    try:
        model = LTXImageToVideoPipeline.from_single_file(
            str(model_file),
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            local_files_only=False,
        )

        if hasattr(model, "remove_all_hooks"):
            try:
                model.remove_all_hooks()
            except Exception:
                pass

        clear_cuda_cache()

    except Exception as exc:
        raise gr.Error(
            "Failed to load the local LTX-Video checkpoint.\n\n"
            f"Checkpoint:\n{model_file}\n\n"
            "If this is the new FP8 checkpoint, make sure your diffusers version supports it.\n"
            "Also make sure tokenizer/ and text_encoder/ folders exist locally."
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
    pipe_key = current_key
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
        raise gr.Error("LOW_VRAM_MODE requires tokenizer and text_encoder.")

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


def run_pipeline(
    model,
    image,
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    steps,
    generator,
    guidance_scale,
    low_vram_mode,
):
    if low_vram_mode:
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = encode_prompts_on_cpu(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_sequence_length=128,
        )

        execution_device = getattr(model, "_execution_device", torch.device(DEVICE))

        prompt_embeds = prompt_embeds.to(device=execution_device, dtype=torch.float16)
        negative_prompt_embeds = negative_prompt_embeds.to(device=execution_device, dtype=torch.float16)
        prompt_attention_mask = prompt_attention_mask.to(device=execution_device)
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(device=execution_device)

        return model(
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
        )

    return model(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
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


def generate_video(image_path, prompt, quality_preset, low_vram_mode, duration, resolution, seed, steps, guidance_scale):
    ensure_folders()

    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt / scenario.")

    quality_preset = str(quality_preset or "low_vram")
    low_vram_mode = bool(low_vram_mode)

    # Hard safety: RTX 3060 Ti 8GB and weaker must not use full GPU mode.
    if should_force_low_vram():
        low_vram_mode = True

    width, height = parse_resolution(resolution)
    duration = int(duration)
    steps = int(steps)
    guidance_scale = float(guidance_scale)
    num_frames = frame_count(duration)

    model_file = get_configured_model_file()
    model_file_label = str(model_file) if model_file else "not found"

    settings = {
        "model_file": model_file_label,
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
            result = run_pipeline(
                model=model,
                image=image,
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=width,
                height=height,
                num_frames=num_frames,
                steps=steps,
                generator=generator,
                guidance_scale=guidance_scale,
                low_vram_mode=low_vram_mode,
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
        gr.Markdown(
            "Image-to-video: upload an image, write a prompt, then click Generate. "
            "For RTX 3060 Ti 8GB use low_vram, 512x512, 3 sec, 8-16 steps."
        )

        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Input image", type="filepath")
                prompt = gr.Textbox(
                    label="Prompt / scenario",
                    lines=5,
                    placeholder="Describe simple motion. Example: The robot gently moves its head and raises one hand. Static camera.",
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
