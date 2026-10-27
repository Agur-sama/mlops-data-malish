from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from clearml import Dataset


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачивание и проверка ClearML Dataset по ID")
    parser.add_argument("ids", type=Path, nargs="?", default=Path("clearml_ids.json"))
    parser.add_argument("--download-root", type=Path, default=Path("clearml_downloads"))
    parser.add_argument("--output", type=Path, default=Path("clearml_verification.json"))
    args = parser.parse_args()

    identifiers = json.loads(args.ids.read_text(encoding="utf-8"))
    if args.download_root.exists():
        shutil.rmtree(args.download_root)
    args.download_root.mkdir(parents=True)

    result: dict[str, object] = {}
    for label, dataset_id in identifiers.items():
        dataset = Dataset.get(dataset_id=dataset_id)
        target = args.download_root / label.lower()
        dataset.get_mutable_local_copy(target_folder=target, overwrite=False)
        files = []
        for csv_path in sorted(target.rglob("*.csv")):
            frame = pd.read_csv(csv_path)
            files.append(
                {
                    "path": str(csv_path.relative_to(target)),
                    "sha256": sha256_file(csv_path),
                    "shape": list(frame.shape),
                    "columns": list(frame.columns),
                }
            )
        dependency_graph = dataset.get_dependency_graph()
        result[label] = {
            "dataset_id": dataset.id,
            "version": dataset.version,
            "parent_ids": dependency_graph.get(dataset.id, []),
            "dependency_graph": dependency_graph,
            "all_files": dataset.list_files(),
            "added_files": dataset.list_added_files(),
            "modified_files": dataset.list_modified_files(),
            "removed_files": dataset.list_removed_files(),
            "downloaded_files": files,
        }

    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
