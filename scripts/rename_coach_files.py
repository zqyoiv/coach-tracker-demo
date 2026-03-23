from pathlib import Path


ROOT = Path(r"c:\Users\vioyq\Desktop\Coach_Tracker\coach-tracker-demo")
TARGETS = ["coach-3", "coach-4", "coach-5"]


def main() -> None:
    for name in TARGETS:
        folder = ROOT / name
        if not folder.exists():
            print(f"Skip missing {folder}")
            continue

        # Rename files/dirs containing coach-2 token, deepest paths first.
        paths = sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True)
        renamed = 0
        for p in paths:
            if "coach-2" not in p.name:
                continue
            new_name = p.name.replace("coach-2", name)
            new_path = p.with_name(new_name)
            if new_path.exists():
                continue
            p.rename(new_path)
            renamed += 1
        print(f"{name}: renamed {renamed} paths")


if __name__ == "__main__":
    main()

