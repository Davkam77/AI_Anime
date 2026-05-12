from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "Lightricks/LTX-Video"
local_dir = Path("models/ltx-video-fp8")
local_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    allow_patterns=[
        "tokenizer/*",
        "text_encoder/*",
    ],
)

print("DONE: tokenizer and text_encoder downloaded")
