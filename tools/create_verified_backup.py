import concurrent.futures
import datetime
import hashlib
import os
import shutil
from typing import Dict, List, Optional, Tuple


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_and_hash(src: str, dest_dir: str) -> Tuple[str, str, int, str]:
    dest_path = os.path.join(dest_dir, os.path.basename(src))
    shutil.copy2(src, dest_path)
    size = os.path.getsize(dest_path)
    digest = _sha256(dest_path)
    return src, dest_path, size, digest


def _parse_manifest_entries(manifest_path: str) -> Dict[str, Tuple[int, str]]:
    entries: Dict[str, Tuple[int, str]] = {}
    in_files = False
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "FILES:":
                in_files = True
                continue
            if not in_files:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            name, size_str, digest = parts
            try:
                size = int(size_str)
            except ValueError:
                continue
            entries[name] = (size, digest)
    return entries


def _verify_manifest(backup_dir: str, manifest_path: str, max_workers: int) -> List[str]:
    errors: List[str] = []
    entries = _parse_manifest_entries(manifest_path)
    if not entries:
        return ["Manifest parse failed: no file entries found."]

    def _verify_one(name: str, size_expected: int, digest_expected: str) -> Optional[str]:
        path = os.path.join(backup_dir, name)
        if not os.path.exists(path):
            return f"Missing file in backup: {name}"
        size_actual = os.path.getsize(path)
        if size_actual != size_expected:
            return f"Size mismatch for {name}: {size_actual} != {size_expected}"
        digest_actual = _sha256(path)
        if digest_actual != digest_expected:
            return f"Checksum mismatch for {name}"
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_verify_one, name, size, digest): name
            for name, (size, digest) in entries.items()
        }
        for future in concurrent.futures.as_completed(future_map):
            err = future.result()
            if err:
                errors.append(err)
    return errors


def create_verified_backup():
    # 1. Setup Verified Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backups/GOLDEN_STATE_VERIFIED_{timestamp}"
    
    if not os.path.exists(backup_name):
        os.makedirs(backup_name)
    
    # 2. Critical System Files to Freeze
    files_to_save = [
        "config/generated_strategies.json",  # The Portfolio (Gen 12 + Machine Gun)
        "execution/engine.py",               # The Brain (Unified Velocity Ranking)
        "optimization/evolution.py",         # The Trainer
        "optimization/optimizer.py",         # The Loop
        "app.py",                            # The Dashboard
        "models/apex_neural_v6.pkl",          # Latest neural model
        "data/paper_portfolio.json",         # Live paper portfolio state
    ]

    required_files = {
        "config/generated_strategies.json",
        "execution/engine.py",
        "models/apex_neural_v6.pkl",
        "data/paper_portfolio.json",
    }
    
    print(f"🔐 CREATING VERIFIED BACKUP: {backup_name}...")

    existing_files = [f for f in files_to_save if os.path.exists(f)]
    missing_files = [f for f in files_to_save if f not in existing_files]
    for f in missing_files:
        print(f"   ⚠️ Warning: {f} not found!")

    missing_required = [f for f in required_files if not os.path.exists(f)]
    if missing_required:
        raise SystemExit(f"❌ Missing required files: {', '.join(missing_required)}")

    max_workers = min(8, (os.cpu_count() or 4))
    manifest_entries: List[Tuple[str, str, int, str]] = []
    copy_errors: List[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_copy_and_hash, f, backup_name): f
            for f in existing_files
        }
        for future in concurrent.futures.as_completed(future_map):
            src = future_map[future]
            try:
                src, dest, size, digest = future.result()
            except Exception as exc:
                copy_errors.append(f"{src}: {exc}")
                continue
            manifest_entries.append((src, dest, size, digest))
            print(f"   ✅ Secured: {src}")

    if copy_errors:
        raise SystemExit("❌ Backup copy failed:\n" + "\n".join(copy_errors))
            
    # 3. Create Manifest with Performance Evidence
    manifest_path = f"{backup_name}/MANIFEST.txt"
    with open(manifest_path, "w") as f:
        f.write("APEX SNIPER - VERIFIED GOLDEN STATE\n")
        f.write("==================================================\n")
        f.write(f"Date: {timestamp}\n")
        f.write("Verification Source: 2025-11-30T22-40_export.csv\n\n")
        
        f.write("STRATEGY 1: THE SHIELD (Gen 12 Sniper)\n")
        f.write("   - Performance: 48.96% CAGR\n")
        f.write("   - Avg Profit:  18.35%\n")
        f.write("   - Trades:      49\n")
        f.write("   - Logic:       CCI < 0, BB Width > 0.1 (Deep Value)\n\n")
        
        f.write("STRATEGY 2: THE SWORD (Machine Gun)\n")
        f.write("   - Performance: 45.98% CAGR\n")
        f.write("   - Avg Profit:  1.61%\n")
        f.write("   - Trades:      432\n")
        f.write("   - Logic:       ADX < RSI14, Profit Target 1.06 (Velocity)\n")

        f.write("\nFILES:\n")
        for src, dest, size, digest in sorted(manifest_entries, key=lambda x: os.path.basename(x[0]).lower()):
            name = os.path.basename(dest)
            f.write(f"{name}\t{size}\t{digest}\n")

    verify_errors = _verify_manifest(backup_name, manifest_path, max_workers)
    if verify_errors:
        raise SystemExit("❌ Manifest verification failed:\n" + "\n".join(verify_errors))
        
    print("\n✅ BACKUP COMPLETE. This state is now frozen in time.")


if __name__ == "__main__":
    create_verified_backup()
