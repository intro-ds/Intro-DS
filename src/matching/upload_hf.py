"""Upload the dataset folder to the Hugging Face Hub as a public dataset.

Authentication: set the HF_TOKEN environment variable (write token) or run `hf auth login` first.

Usage:
    python -m src.matching.upload_hf --repo-id <username>/<dataset-name>
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

from src.matching.train_ranker import DEFAULT_PROCESSED_DIR


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Upload the dataset folder to Hugging Face")
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/intro-ds-arxiv-matching")
    parser.add_argument("--folder", type=Path, default=DEFAULT_PROCESSED_DIR / "hf_dataset")
    parser.add_argument("--private", action="store_true", help="Create a private repo (the submission needs a public one)")
    args = parser.parse_args()

    if not (args.folder / "README.md").exists():
        raise SystemExit(f"{args.folder}/README.md not found: run export_hf_dataset first.")

    api = HfApi(token=os.environ.get("HF_TOKEN"))  # None -> use the cached `hf auth login`
    print("Logged in as:", api.whoami()["name"])
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.folder),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add arXiv parsing and reference matching dataset (6 Parquet tables)",
    )
    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()