import shutil
from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def move_files(base: Path, patterns: list[str], dest: Path, exclude: set[str]) -> list[tuple[Path, Path]]:
    """Move files matching any pattern into dest, skipping exclusions."""
    moved = []
    seen = set()
    for pattern in patterns:
        for file_path in base.glob(pattern):
            if not file_path.is_file():
                continue
            if file_path.name in exclude:
                continue
            if file_path in seen:
                continue
            seen.add(file_path)
            ensure_dir(dest)
            target = dest / file_path.name
            shutil.move(str(file_path), target)
            moved.append((file_path, dest))
    return moved


def organize_workspace() -> None:
    root = Path.cwd()
    archive_root = root / "_archive" / "root_clutter"
    tools_archive = root / "tools" / "_archive"

    # Do not touch these
    root_exclusions = {
        "app.py",
    }
    tools_exclusions = {
        "create_verified_backup.py",
        "create_restore_point.py",
    }

    root_patterns = [
        "*copy*.py",
        "verify_*.py",
        "debug_*.py",
        "repro_*.py",
        "remove_clones.py",
        "deduplicate_strategies.py",
        "find_bad_data.py",
        "backtest_runner.py",
        "mcp_test_client.js",
        "*.txt",
    ]

    tools_patterns = [
        "fix_*.py",
        "force_*.py",
        "restore_*.py",
        "upgrade_*.py",
        "apply_*.py",
        "backup_*.py",
    ]

    moved_logs = []
    moved_logs.extend(move_files(root, root_patterns, archive_root, root_exclusions))
    tools_dir = root / "tools"
    if tools_dir.exists():
        moved_logs.extend(move_files(tools_dir, tools_patterns, tools_archive, tools_exclusions))

    for src, dest_dir in moved_logs:
        print(f"Moved: {src.name} -> {dest_dir.as_posix()}/")

    print("✅ Cleanup Complete. Workspace is clean. Old files are in _archive.")


if __name__ == "__main__":
    organize_workspace()
