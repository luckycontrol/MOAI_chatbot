from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
    local_dir_use_symlinks=False
)