from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "Lightricks/LTX-Video"
local_dir = Path("models/ltx-video")

snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    allow_patterns=[
        "scheduler/*",
        "transformer/config.json",
        "vae/config.json",
    ],
)

print("DONE: configs downloaded")
