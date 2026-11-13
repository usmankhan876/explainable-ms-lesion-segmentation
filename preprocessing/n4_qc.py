"""
Quality check for N4 bias-field correction.

Compares each skull-stripped volume ("<id>-<modality>_bet.nii.gz") with its N4-corrected version
("<id>-<modality>_n4.nii.gz") using numbers, not only visual inspection:

1. Identical check: if the two volumes are voxel-identical, N4 was NOT applied to that file.
2. Coefficient of variation (CV = std / mean) inside the brain (voxels > 0 in the skull-stripped
   volume). N4 should reduce intensity variation, so a lower CV after N4 suggests it helped.
3. Estimated bias field = before / after (inside the brain). A real bias field is smooth and
   low-frequency; its standard deviation is reported as a rough sanity check.

The verdict thresholds (5% / 0%) are heuristics, not a validated criterion. Treat them as a
screening tool and look at the saved figures as well.

Usage:
    python preprocessing/n4_qc.py --pre_dir data_processed_bet --post_dir N4_Corrected --out_dir n4_qc
Outputs: n4_qc_results.csv and n4_qc_example_*.png in --out_dir.
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

PRE_PATTERN = re.compile(r"^(\d+)-(Flair|T1|T2)_bet\.nii\.gz$")
POST_PATTERN = re.compile(r"^(\d+)-(Flair|T1|T2)_n4\.nii\.gz$")


def index_files(folder: Path, pattern):
    """Map (patient_id, modality) -> path, skipping 'Copy of' duplicates."""
    found = {}
    for f in sorted(folder.glob("*.nii.gz")):
        if "Copy of" in f.name:
            continue
        m = pattern.match(f.name)
        if m:
            found[(m.group(1), m.group(2))] = f
    return found


def check_pair(pre_path, post_path, key, out_dir, plot):
    pid, modality = key
    pre = nib.load(str(pre_path)).get_fdata()
    post = nib.load(str(post_path)).get_fdata()
    row = {"patient": pid, "modality": modality}

    if pre.shape != post.shape:
        row.update(status="shape_mismatch", verdict="SKIPPED")
        print(f"[{pid}-{modality}] SHAPE MISMATCH {pre.shape} vs {post.shape}, skipping")
        return row

    if np.allclose(pre, post, atol=1e-6):
        row.update(status="not_applied", verdict="N4 NOT APPLIED (identical)")
        print(f"[{pid}-{modality}] identical to the skull-stripped volume: N4 was NOT applied.")
        return row

    mask = pre > 0
    if mask.sum() == 0 or pre[mask].mean() == 0 or post[mask].mean() == 0:
        row.update(status="empty_mask", verdict="SKIPPED")
        print(f"[{pid}-{modality}] empty brain mask, skipping")
        return row

    cv_pre = pre[mask].std() / pre[mask].mean()
    cv_post = post[mask].std() / post[mask].mean()
    change = 100 * (cv_pre - cv_post) / cv_pre

    bias = np.full(pre.shape, np.nan)
    bias[mask] = pre[mask] / np.clip(post[mask], 1e-6, None)
    bias_std = float(np.nanstd(bias[mask]))

    if change > 5:
        verdict = "IMPROVED"
    elif change > 0:
        verdict = "SLIGHT IMPROVEMENT"
    else:
        verdict = "NO IMPROVEMENT / WORSE"

    row.update(status="applied", cv_pre=round(cv_pre, 4), cv_post=round(cv_post, 4),
               cv_change_pct=round(change, 2), bias_field_std=round(bias_std, 4), verdict=verdict)
    print(f"[{pid}-{modality}] CV {cv_pre:.4f} -> {cv_post:.4f} ({change:+.2f}%), "
          f"bias-field std {bias_std:.4f} -> {verdict}")

    if plot:
        mid = pre.shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(pre[:, :, mid].T, cmap="gray", origin="lower"); axes[0].set_title("Before N4")
        axes[1].imshow(post[:, :, mid].T, cmap="gray", origin="lower"); axes[1].set_title("After N4")
        im = axes[2].imshow(bias[:, :, mid].T, cmap="jet", origin="lower")
        axes[2].set_title("Estimated bias field\n(should look smooth)")
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        for ax in axes:
            ax.axis("off")
        plt.suptitle(f"Patient {pid}, {modality}")
        plt.tight_layout()
        plt.savefig(out_dir / f"n4_qc_example_{pid}-{modality}.png", dpi=120)
        plt.close(fig)
    return row


def main(args):
    pre_dir, post_dir, out_dir = Path(args.pre_dir), Path(args.post_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_map = index_files(pre_dir, PRE_PATTERN)
    post_map = index_files(post_dir, POST_PATTERN)
    common = sorted(set(pre_map) & set(post_map), key=lambda k: (int(k[0]), k[1]))
    print(f"Matched {len(common)} (patient, modality) pairs "
          f"({len(pre_map)} pre-N4 files, {len(post_map)} N4 files).\n")
    if not common:
        print("No matches. Sample names, pre:", [p.name for p in list(pre_map.values())[:3]],
              "| post:", [p.name for p in list(post_map.values())[:3]])
        return

    rows, plotted = [], 0
    for key in common:
        do_plot = args.n_plots > 0 and plotted < args.n_plots and key[1] == args.plot_modality
        row = check_pair(pre_map[key], post_map[key], key, out_dir, do_plot)
        if do_plot and row["status"] == "applied":
            plotted += 1
        rows.append(row)

    fields = ["patient", "modality", "status", "cv_pre", "cv_post", "cv_change_pct", "bias_field_std", "verdict"]
    with open(out_dir / "n4_qc_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    applied = [r for r in rows if r["status"] == "applied"]
    print("\n========== SUMMARY ==========")
    print(f"Pairs compared          : {len(rows)}")
    print(f"N4 not applied          : {sum(r['status'] == 'not_applied' for r in rows)}")
    print(f"N4 applied              : {len(applied)}")
    print(f"  Improved              : {sum(r['verdict'] == 'IMPROVED' for r in applied)}")
    print(f"  Slight improvement    : {sum(r['verdict'] == 'SLIGHT IMPROVEMENT' for r in applied)}")
    print(f"  No improvement / worse: {sum(r['verdict'].startswith('NO IMPROVEMENT') for r in applied)}")
    print(f"Results saved to {out_dir / 'n4_qc_results.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quantitative QC of N4 bias-field correction")
    p.add_argument("--pre_dir", required=True, help="folder with *_bet.nii.gz (before N4)")
    p.add_argument("--post_dir", required=True, help="folder with *_n4.nii.gz (after N4)")
    p.add_argument("--out_dir", default="n4_qc", help="where to save the CSV and figures")
    p.add_argument("--n_plots", type=int, default=3, help="number of example figures to save")
    p.add_argument("--plot_modality", default="Flair", choices=["Flair", "T1", "T2"])
    main(p.parse_args())
