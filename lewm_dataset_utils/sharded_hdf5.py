"""Helpers for shard-aware HDF5 dataset layouts."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
STATE_FILENAME = "publish_state.json"


def camera_slug(camera_key: str) -> str:
    return camera_key.replace(".", "_")


def shard_id_str(index: int) -> str:
    return f"shard-{index:05d}"


def shard_dir_relpath(split_name: str, shard_id: str) -> Path:
    return Path(split_name) / shard_id


def shard_camera_relpath(split_name: str, shard_id: str, camera_key: str) -> Path:
    return shard_dir_relpath(split_name, shard_id) / f"{camera_slug(camera_key)}.h5"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_path(root: Path) -> Path:
    return root / MANIFEST_FILENAME


def state_path(root: Path) -> Path:
    return root / STATE_FILENAME


def has_manifest(root: Path) -> bool:
    return manifest_path(root).is_file()


def load_manifest(root: Path) -> dict[str, Any]:
    return load_json_file(manifest_path(root))


def load_state(root: Path) -> dict[str, Any] | None:
    path = state_path(root)
    if not path.is_file():
        return None
    return load_json_file(path)


def iter_manifest_files(
    manifest: dict[str, Any],
    *,
    split: str | None = None,
    camera_key: str | None = None,
    shard_ids: list[str] | None = None,
) -> list[str]:
    selected_shards = set(shard_ids or [])
    files: list[str] = []
    splits = manifest.get("splits", {})
    for split_name, split_payload in splits.items():
        if split and split_name != split:
            continue
        for shard in split_payload.get("shards", []):
            shard_id = str(shard["shard_id"])
            if selected_shards and shard_id not in selected_shards:
                continue
            if camera_key:
                relpath = shard.get("camera_files", {}).get(camera_key)
                if relpath is not None:
                    files.append(str(relpath))
                continue
            files.extend(str(path) for path in shard.get("camera_files", {}).values())
    return files


def resolve_manifest_hdf5_files(
    root: Path,
    *,
    split: str | None = None,
    camera_key: str | None = None,
    shard_ids: list[str] | None = None,
) -> list[Path]:
    manifest = load_manifest(root)
    return [root / relpath for relpath in iter_manifest_files(
        manifest,
        split=split,
        camera_key=camera_key,
        shard_ids=shard_ids,
    )]


def remote_file_size_map(repo_info: Any) -> dict[str, int]:
    size_map: dict[str, int] = {}
    for sibling in getattr(repo_info, "siblings", []) or []:
        size = getattr(sibling, "size", None)
        if size is None:
            lfs = getattr(sibling, "lfs", None)
            if lfs is not None:
                size = getattr(lfs, "size", None)
        if size is None:
            continue
        size_map[str(sibling.rfilename)] = int(size)
    return size_map
