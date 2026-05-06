from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "Lightricks/LTX-Video"
local_dir = Path("models/ltx-video")
local_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    allow_patterns=[
        "ltx-video-2b-v0.9.safetensors",
        "model_index.json",
        "tokenizer/*",
        "text_encoder/*",
    ],
)

print("DONE: model files downloaded to", local_dir.resolve())
