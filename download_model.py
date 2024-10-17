from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1",
    local_dir_use_symlinks=False
)