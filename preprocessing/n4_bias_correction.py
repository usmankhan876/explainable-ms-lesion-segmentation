"""
Step 2 of preprocessing: N4 bias-field correction (3D) with SimpleITK.

Reads the HD-BET skull-stripped volumes ("<id>-<modality>_bet.nii.gz") from --input_dir and writes
the corrected volumes ("<id>-<modality>_n4.nii.gz") to --output_dir. Input files are never modified.
Duplicate files whose name contains "Copy of" (e.g. Google Drive copies) are skipped.

Because the input is already skull-stripped, a foreground mask is built with Otsu thresholding so
that N4 estimates the bias field inside the brain only.

Usage:
    pip install SimpleITK
    python preprocessing/n4_bias_correction.py \
        --input_dir data_processed_bet --output_dir N4_Corrected

N4ITK: Tustison et al., "N4ITK: Improved N3 bias correction", IEEE Trans. Med. Imaging, 2010.
"""
import argparse
import re
from pathlib import Path

import SimpleITK as sitk

NAME_PATTERN = re.compile(r"^(\d+)-(Flair|T1|T2)_bet\.nii\.gz$")


def get_patient_files(input_dir: Path, modalities):
    """Return sorted (patient_id, modality, path) for files named '<id>-<modality>_bet.nii.gz'."""
    items = []
    for f in sorted(input_dir.glob("*_bet.nii.gz")):
        if "Copy of" in f.name:
            continue
        m = NAME_PATTERN.match(f.name)
        if not m:
            print(f"  Skipping unrecognized filename: {f.name}")
            continue
        patient_id, modality = m.groups()
        if modality in modalities:
            items.append((patient_id, modality, f))
    return sorted(items, key=lambda x: (int(x[0]), x[1]))


def run_n4(input_path: Path, output_path: Path):
    """Apply N4 bias-field correction to one volume."""
    img = sitk.ReadImage(str(input_path), sitk.sitkFloat32)

    # Foreground mask (non-zero brain tissue) so that N4 only corrects inside the brain.
    mask = sitk.OtsuThreshold(img, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(img, mask)
    sitk.WriteImage(corrected, str(output_path))


def main(args):
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = get_patient_files(input_dir, args.modalities)
    print(f"Found {len(files)} files to process (modalities: {', '.join(args.modalities)}).\n")

    for patient_id, modality, in_path in files:
        out_name = f"{patient_id}-{modality}_n4.nii.gz"
        out_path = output_dir / out_name

        if out_path.exists() and not args.overwrite:
            print(f"  [skip] {out_name} already exists")
            continue

        print(f"  [N4] Patient {patient_id}, {modality} ...")
        try:
            run_n4(in_path, out_path)
            print(f"       -> saved {out_name}")
        except Exception as e:  # keep going if a single volume fails
            print(f"       -> FAILED: {e}")

    print(f"\nDone. N4-corrected files saved to: {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="N4 bias-field correction with SimpleITK")
    p.add_argument("--input_dir", required=True, help="folder with *_bet.nii.gz files (HD-BET output)")
    p.add_argument("--output_dir", default="N4_Corrected", help="output folder")
    p.add_argument("--modalities", nargs="+", default=["Flair", "T1", "T2"],
                   choices=["Flair", "T1", "T2"], help="modalities to process (default: all three)")
    p.add_argument("--overwrite", action="store_true", help="re-run files that already exist")
    main(p.parse_args())
