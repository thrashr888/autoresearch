"""
Materialize ANE-friendly token streams from the existing autoresearch cache.

This is a one-time preprocessing step for the ANE-backed fork:

1. Run `uv run prepare.py` to download parquet shards and train the tokenizer.
2. Run `uv run prepare_ane.py` to convert that cache into uint16 token streams
   that the ANE trainer can mmap directly.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prepare import CACHE_DIR, TOKENIZER_DIR, VAL_FILENAME, Tokenizer, list_parquet_files

ANE_CACHE_DIR = Path(CACHE_DIR) / "ane"
TRAIN_TOKENS_PATH = ANE_CACHE_DIR / "train_tokens_u16.bin"
VAL_TOKENS_PATH = ANE_CACHE_DIR / "val_tokens_u16.bin"
MANIFEST_PATH = ANE_CACHE_DIR / "manifest.json"

DEFAULT_TRAIN_TOKENS = 10_000_000
DEFAULT_VAL_TOKENS = 1_000_000
TOKENIZER_BATCH_SIZE = 128


def _iter_text_batches(parquet_paths: list[str], batch_size: int):
    for filepath in parquet_paths:
        parquet_file = pq.ParquetFile(filepath)
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_idx)
            texts = row_group.column("text").to_pylist()
            for start in range(0, len(texts), batch_size):
                yield texts[start : start + batch_size]


def _write_stream(
    output_path: Path,
    parquet_paths: list[str],
    tokenizer: Tokenizer,
    target_tokens: int,
) -> dict[str, object]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bos_token_id = tokenizer.get_bos_token_id()
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"Tokenizer vocab_size={vocab_size} exceeds uint16 token capacity")

    total_tokens = 0
    total_docs = 0
    shards_used: list[str] = []

    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            for filepath in parquet_paths:
                shards_used.append(Path(filepath).name)
                for text_batch in _iter_text_batches([filepath], TOKENIZER_BATCH_SIZE):
                    token_lists = tokenizer.encode(text_batch, prepend=bos_token_id)
                    for ids in token_lists:
                        remaining = target_tokens - total_tokens
                        if remaining <= 0:
                            break
                        token_array = np.asarray(ids[:remaining], dtype=np.uint16)
                        token_array.tofile(tmp)
                        total_tokens += int(token_array.size)
                        total_docs += 1
                    if total_tokens >= target_tokens:
                        break
                if total_tokens >= target_tokens:
                    break
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    tmp_path.replace(output_path)
    return {
        "path": str(output_path),
        "tokens": total_tokens,
        "documents": total_docs,
        "shards": shards_used,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ANE token streams for autoresearch")
    parser.add_argument("--force", action="store_true", help="Rebuild token streams even if they already exist")
    parser.add_argument(
        "--train-tokens",
        type=int,
        default=DEFAULT_TRAIN_TOKENS,
        help=f"Number of train tokens to materialize (default: {DEFAULT_TRAIN_TOKENS:,})",
    )
    parser.add_argument(
        "--val-tokens",
        type=int,
        default=DEFAULT_VAL_TOKENS,
        help=f"Number of validation tokens to materialize (default: {DEFAULT_VAL_TOKENS:,})",
    )
    args = parser.parse_args()

    tokenizer_dir = Path(TOKENIZER_DIR)
    if not tokenizer_dir.exists():
        raise SystemExit("Tokenizer cache is missing. Run `uv run prepare.py` first.")

    parquet_paths = list_parquet_files()
    if not parquet_paths:
        raise SystemExit("No parquet shards found. Run `uv run prepare.py` first.")

    train_parquet_paths = [path for path in parquet_paths if not path.endswith(VAL_FILENAME)]
    val_parquet_paths = [path for path in parquet_paths if path.endswith(VAL_FILENAME)]
    if not train_parquet_paths:
        raise SystemExit("No training shards found in the autoresearch cache.")
    if not val_parquet_paths:
        raise SystemExit("Pinned validation shard is missing from the autoresearch cache.")

    if (
        not args.force
        and TRAIN_TOKENS_PATH.exists()
        and VAL_TOKENS_PATH.exists()
        and MANIFEST_PATH.exists()
    ):
        print(f"ANE cache already exists at {ANE_CACHE_DIR}")
        print(f"Train stream: {TRAIN_TOKENS_PATH}")
        print(f"Val stream:   {VAL_TOKENS_PATH}")
        print(f"Manifest:     {MANIFEST_PATH}")
        return

    tokenizer = Tokenizer.from_directory()

    print(f"Writing ANE token streams into {ANE_CACHE_DIR}")
    train_info = _write_stream(TRAIN_TOKENS_PATH, train_parquet_paths, tokenizer, args.train_tokens)
    val_info = _write_stream(VAL_TOKENS_PATH, val_parquet_paths, tokenizer, args.val_tokens)

    manifest = {
        "cache_dir": str(ANE_CACHE_DIR),
        "tokenizer_dir": str(tokenizer_dir),
        "vocab_size": tokenizer.get_vocab_size(),
        "bos_token_id": tokenizer.get_bos_token_id(),
        "train": train_info,
        "val": val_info,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Train tokens: {train_info['tokens']:,} -> {TRAIN_TOKENS_PATH}")
    print(f"Val tokens:   {val_info['tokens']:,} -> {VAL_TOKENS_PATH}")
    print(f"Manifest:     {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
