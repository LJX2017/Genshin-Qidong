"""
Copy a small subset of lightmap binaries for quick inspection.

By default this copies all lightmap files referenced by the first 5 entries
in config.json into ./lightmap_sample (without modifying the original data).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy sample lightmap files for the first N entries.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../Data/Data_HPRC"),
        help="Path to the dataset root containing config.json and Data/ folder.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of lightmap entries from config.json to sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./lightmap_sample"),
        help="Destination folder for copied files (created if missing).",
    )
    parser.add_argument(
        "--include-masks",
        action="store_true",
        help="Also copy mask files referenced by the sampled entries.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_files(lightmap_list: list[dict], limit: int, include_masks: bool) -> set[str]:
    selected = lightmap_list[:limit]
    filenames: set[str] = set()
    for entry in selected:
        filenames.update(entry.get("lightmaps", {}).values())
        if include_masks:
            filenames.update(entry.get("masks", {}).values())
    return filenames


def copy_files(filenames: set[str], data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(filenames):
        src = data_dir / name
        if not src.exists():
            print(f"[WARN] Missing source file, skipping: {src}")
            continue
        dst = output_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")


def main() -> None:
    args = parse_args()
    config_path = args.dataset_root / "config.json"
    data_dir = args.dataset_root / "Data"

    cfg = load_config(config_path)
    lightmap_list = cfg.get("lightmap_list", [])
    if not lightmap_list:
        raise ValueError(f"No lightmap_list found in {config_path}")

    filenames = collect_files(lightmap_list, args.limit, args.include_masks)
    print(f"Found {len(filenames)} unique files to copy from first {args.limit} entries.")
    copy_files(filenames, data_dir, args.output)


if __name__ == "__main__":
    main()
