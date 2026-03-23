from pathlib import Path
import shutil


ROOT = Path(r"c:\Users\vioyq\Desktop\Coach_Tracker\coach-tracker-demo")
SRC = ROOT / "coach-2"
TARGETS = ["coach-3", "coach-4", "coach-5"]

# Text-like files where string replacement is safe.
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".csv", ".txt", ".md", ".json"}


def replace_in_text_files(folder: Path, src_token: str, dst_token: str) -> int:
    changed = 0
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = raw.replace(src_token, dst_token).replace(src_token.title(), dst_token.title())
        if new != raw:
            path.write_text(new, encoding="utf-8")
            changed += 1
    return changed


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source folder missing: {SRC}")

    for target_name in TARGETS:
        target = ROOT / target_name
        if target.exists():
            raise SystemExit(f"Target already exists, aborting: {target}")

    for target_name in TARGETS:
        target = ROOT / target_name
        shutil.copytree(SRC, target)
        changed_files = replace_in_text_files(target, "coach-2", target_name)
        print(f"Created {target_name}; updated {changed_files} text files.")


if __name__ == "__main__":
    main()

