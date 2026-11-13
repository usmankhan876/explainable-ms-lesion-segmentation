"""
Step 1 of preprocessing: 3D skull stripping with HD-BET.

Runs the HD-BET command-line tool on every NIfTI volume (*.nii) found under --raw_dir and
writes <name>_bet.nii.gz to --out_dir. Lesion masks (paths containing "LesionSeg") are skipped.

This is a cleaned-up version of notebooks/01_skull_stripping_hdbet.ipynb. The behaviour matches
the notebook: GPU inference with test-time augmentation disabled (--disable_tta) by default.

Usage:
    pip install hd-bet
    python preprocessing/skull_strip.py \
        --raw_dir /path/to/multiplesclerosis-60-patients-data \
        --out_dir data_processed_bet

HD-BET: Isensee et al., "Automated brain extraction of multisequence MRI using artificial
neural networks", Human Brain Mapping, 2019.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def find_volumes(raw_dir: Path, pattern: str):
    """All matching volumes under raw_dir, excluding lesion segmentation masks."""
    files = sorted(raw_dir.rglob(pattern))
    return [f for f in files if "LesionSeg" not in str(f)]


def main(args):
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    volumes = find_volumes(raw_dir, args.pattern)
    print(f"Found {len(volumes)} volumes for processing in {raw_dir}")
    if not volumes:
        sys.exit("No volumes found. Check --raw_dir and --pattern.")

    # Output names use only the file name (as in the original notebook), so two files with
    # the same name in different folders would overwrite each other.
    names = [f.name for f in volumes]
    if len(set(names)) != len(names):
        print("WARNING: duplicate file names found; some outputs will be overwritten.")

    for i, f in enumerate(volumes, start=1):
        base = f.name[: -len(".nii")] if f.name.endswith(".nii") else f.stem
        out_file = out_dir / f"{base}_bet.nii.gz"
        if args.skip_existing and out_file.exists():
            print(f"[{i}/{len(volumes)}] Skipping (exists): {out_file}")
            continue

        print(f"[{i}/{len(volumes)}] Processing {f} -> {out_file}")
        cmd = ["hd-bet", "-i", str(f), "-o", str(out_file), "-device", args.device]
        if not args.tta:
            cmd.append("--disable_tta")
        subprocess.run(cmd, check=True)

    print(f"Done. Skull-stripped files (*_bet.nii.gz) are in: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Skull stripping with HD-BET")
    p.add_argument("--raw_dir", required=True, help="folder containing the raw .nii files")
    p.add_argument("--out_dir", default="data_processed_bet", help="output folder")
    p.add_argument("--pattern", default="*.nii", help="file pattern, e.g. '*FLAIR*.nii'")
    p.add_argument("--device", default="cuda", help="cuda, cpu, or mps")
    p.add_argument("--tta", action="store_true", help="enable test-time augmentation (slower)")
    p.add_argument("--skip_existing", action="store_true", help="skip files already processed")
    main(p.parse_args())
