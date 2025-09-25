from pathlib import Path
from labutils import run_with_notify
from opm_batch import process_root

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Batch OPM summary")
    ap.add_argument("root", type=str, help="Root data folder")
    ap.add_argument("--title", type=str, default="OPM Batch Task", help="Notification title")
    args = ap.parse_args()

    run_with_notify(process_root, Path(args.root), title=args.title)
