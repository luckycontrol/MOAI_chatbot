from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/gemma-2-2b-it",
    local_dir_use_symlinks=False
)