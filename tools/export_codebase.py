"""
Export the project source into a single text file for auditing.

- Includes root-level Python files and code under execution/, strategies/, simulation/, config/, data/ (data is .py only).
- Excludes backups/, tools/, .venv/, __pycache__/, .git/ and their children.
- Writes formatted output with file headers/footers to codebase_audit.txt in project root.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple


# Directories to include (relative to repo root)
INCLUDE_DIRS: Tuple[str, ...] = (
    ".",  # root files like app.py
    "execution",
    "strategies",
    "simulation",
    "config",
    "data",
)

# Directories to skip entirely
EXCLUDE_DIRS: Set[str] = {"backups", "tools", ".venv", "__pycache__", ".git"}

# Extensions to include for most folders (config may have json/yaml)
GENERAL_EXTS: Set[str] = {".py", ".json", ".yaml", ".yml", ".ini", ".cfg", ".toml", ".md", ".txt", ".csv"}
# data/ folder should only contribute .py files
DATA_EXTS: Set[str] = {".py"}


def should_skip_dir(dir_name: str) -> bool:
    return dir_name in EXCLUDE_DIRS


def is_allowed_file(path: Path, allowed_exts: Set[str]) -> bool:
    # Skip the audit output itself
    if path.name == "codebase_audit.txt":
        return False
    return path.is_file() and path.suffix.lower() in allowed_exts


def gather_root_files(root: Path) -> List[Path]:
    allowed = []
    for item in root.iterdir():
        if item.is_dir() or should_skip_dir(item.name):
            continue
        if is_allowed_file(item, GENERAL_EXTS):
            allowed.append(item)
    return sorted(allowed)


def gather_dir_files(root: Path, rel_dir: str, allowed_exts: Set[str]) -> List[Path]:
    collected: List[Path] = []
    base = (root / rel_dir).resolve()
    if not base.exists():
        return collected

    for current, dirs, files in os_walk_filtered(base):
        for fname in files:
            path = current / fname
            if is_allowed_file(path, allowed_exts):
                collected.append(path)
    return sorted(collected)


def os_walk_filtered(base: Path) -> Iterable[Tuple[Path, List[str], List[str]]]:
    """Wrapper around os.walk that prunes excluded directories early."""
    import os

    for current, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]
        yield Path(current), dirs, files


def write_export(file_paths: List[Path], root: Path, output_path: Path) -> None:
    lines: List[str] = []
    for path in file_paths:
        rel = path.relative_to(root)
        lines.append(f"===== START FILE: {rel} =====")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="replace")
        lines.append(content.rstrip("\n"))
        lines.append(f"===== END FILE: {rel} =====")
        lines.append("")  # spacer

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    output_path = root / "codebase_audit.txt"

    files: List[Path] = []
    # Root files first
    files.extend(gather_root_files(root))

    for rel_dir in INCLUDE_DIRS[1:]:
        allowed_exts = DATA_EXTS if rel_dir == "data" else GENERAL_EXTS
        files.extend(gather_dir_files(root, rel_dir, allowed_exts))

    write_export(files, root, output_path)
    print(f"Wrote {len(files)} files to {output_path}")


if __name__ == "__main__":
    main()
