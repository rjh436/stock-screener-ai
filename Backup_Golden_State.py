import datetime
import os
import shutil
from typing import Iterable, List, Set


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _iter_files(src_dir: str, exclude_dirs: Set[str] | None = None) -> Iterable[str]:
    for root, dirs, files in os.walk(src_dir):
        if exclude_dirs:
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for filename in files:
            yield os.path.join(root, filename)


def _copy_file(src_path: str, repo_root: str, backup_root: str, manifest: List[str]) -> None:
    rel_path = os.path.relpath(src_path, start=repo_root)
    dest_path = os.path.join(backup_root, rel_path)
    _ensure_dir(os.path.dirname(dest_path))
    shutil.copy2(src_path, dest_path)
    manifest.append(rel_path)
    print(f"✅ Backed up: {rel_path}")


def _copy_directory(
    src_dir: str,
    repo_root: str,
    backup_root: str,
    manifest: List[str],
    exclude_dirs: Set[str] | None = None,
) -> None:
    for src_path in _iter_files(src_dir, exclude_dirs=exclude_dirs):
        _copy_file(src_path, repo_root, backup_root, manifest)


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(repo_root, "backups", f"V10_STABLE_{timestamp}")

    _ensure_dir(backup_root)

    files_to_copy = [
        "app.py",
        "Run_Screener.command",
        "requirements.txt",
    ]
    dirs_to_copy = [
        "config",
        "execution",
        "strategies",
        "simulation",
    ]

    manifest: List[str] = []
    copy_errors: List[str] = []
    missing_items: List[str] = []

    print(f"🔒 Creating Golden State backup: {backup_root}")

    for rel_path in files_to_copy:
        src_path = os.path.join(repo_root, rel_path)
        if not os.path.exists(src_path):
            missing_items.append(rel_path)
            continue
        try:
            _copy_file(src_path, repo_root, backup_root, manifest)
        except Exception as exc:
            copy_errors.append(f"{rel_path}: {exc}")

    for rel_dir in dirs_to_copy:
        src_dir = os.path.join(repo_root, rel_dir)
        if not os.path.isdir(src_dir):
            missing_items.append(rel_dir + "/")
            continue
        try:
            _copy_directory(src_dir, repo_root, backup_root, manifest)
        except Exception as exc:
            copy_errors.append(f"{rel_dir}/: {exc}")

    data_dir = os.path.join(repo_root, "data")
    if os.path.isdir(data_dir):
        try:
            _copy_directory(
                data_dir,
                repo_root,
                backup_root,
                manifest,
                exclude_dirs={"cache"},
            )
        except Exception as exc:
            copy_errors.append(f"data/: {exc}")
    else:
        missing_items.append("data/")

    if missing_items:
        print("⚠️ Missing items (skipped): " + ", ".join(missing_items))

    if copy_errors:
        for err in copy_errors:
            print(f"❌ Copy failed: {err}")
        raise SystemExit(1)

    manifest_path = os.path.join(backup_root, "MANIFEST.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("GOLDEN STATE BACKUP MANIFEST\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("FILES:\n")
        for rel_path in sorted(manifest):
            f.write(rel_path + "\n")

    print(f"✅ Backup complete. Location: {backup_root}")


if __name__ == "__main__":
    main()
