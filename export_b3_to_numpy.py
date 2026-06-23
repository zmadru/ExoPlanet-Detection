#!/usr/bin/env python
"""Export B3 pickle light curves to compressed NumPy (.npz) files.

Reads ``all_data_2025_B3/*.pickle`` (``LightCurveWaveletGlobalLocalCollection``)
and writes one ``.npz`` per sample with the level-0 flux series and optional
wavelet coefficients. No torch / DirectML required.

Example:
    python export_b3_to_numpy.py --input-dir all_data_2025_B3 --output-dir all_data_2025_B3_npz
    python export_b3_to_numpy.py --with-wavelet --limit 10
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lib.LCWavelet import LightCurveWaveletGlobalLocalCollection


def _flux(lc_fold) -> np.ndarray:
    return np.asarray(lc_fold._light_curve.flux.value, dtype=np.float32)


def _stack_wavelet(wl_collection) -> tuple[np.ndarray, np.ndarray]:
    approx = np.stack([level[0] for level in wl_collection], axis=0).astype(np.float32)
    detail = np.stack([level[1] for level in wl_collection], axis=0).astype(np.float32)
    return approx, detail


def pickle_to_npz(lc: LightCurveWaveletGlobalLocalCollection, with_wavelet: bool) -> dict:
    """Build a dict of arrays ready for ``np.savez_compressed``."""
    data = {
        "global_odd": _flux(lc.pliegue_impar_global),
        "global_even": _flux(lc.pliegue_par_global),
        "local_odd": _flux(lc.pliegue_impar_local),
        "local_even": _flux(lc.pliegue_par_local),
        "kepid": np.int64(lc.kepler_id),
        "kepler_name": np.asarray(lc.headers.get("Kepler_name", "")),
        "koi_class": np.asarray(lc.headers.get("class", "")),
    }

    if with_wavelet:
        for prefix, fold in (
            ("global_odd", lc.pliegue_impar_global),
            ("global_even", lc.pliegue_par_global),
            ("local_odd", lc.pliegue_impar_local),
            ("local_even", lc.pliegue_par_local),
        ):
            wl_a, wl_d = _stack_wavelet(fold._lc_w_collection)
            data[f"{prefix}_wl_a"] = wl_a
            data[f"{prefix}_wl_d"] = wl_d
        levels = lc.headers.get("levels", lc.levels if hasattr(lc, "levels") else [])
        data["wavelet_levels"] = np.asarray(levels, dtype=np.int32)

    return data


def export_sample(
    pickle_path: Path,
    output_dir: Path,
    with_wavelet: bool,
    overwrite: bool,
) -> dict:
    out_path = output_dir / (pickle_path.stem + ".npz")
    if out_path.exists() and not overwrite:
        return {"file": out_path.name, "skipped": True}

    lc = LightCurveWaveletGlobalLocalCollection.from_pickle(str(pickle_path))
    arrays = pickle_to_npz(lc, with_wavelet=with_wavelet)
    np.savez_compressed(out_path, **arrays)

    return {
        "file": out_path.name,
        "kepid": int(arrays["kepid"]),
        "kepler_name": str(arrays["kepler_name"]),
        "class": str(arrays["koi_class"]),
        "global_len": int(arrays["global_odd"].shape[0]),
        "local_len": int(arrays["local_odd"].shape[0]),
        "n_wl_levels": int(arrays["global_odd_wl_a"].shape[0]) if with_wavelet else 0,
        "skipped": False,
    }


def write_index(rows: list[dict], index_path: Path) -> None:
    fieldnames = ["file", "kepid", "kepler_name", "class", "global_len", "local_len", "n_wl_levels"]
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row.get("skipped"):
                continue
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export B3 pickles to NumPy .npz files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("all_data_2025_B3"),
        help="Directory with .pickle files (default: all_data_2025_B3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("all_data_2025_B3_npz"),
        help="Output directory for .npz files (default: all_data_2025_B3_npz)",
    )
    parser.add_argument(
        "--with-wavelet",
        action="store_true",
        help="Also save wavelet approx/detail stacks per branch",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N pickles (for testing)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    pickle_files = sorted(input_dir.glob("*.pickle"))
    if args.limit is not None:
        pickle_files = pickle_files[: args.limit]

    if not pickle_files:
        print(f"No .pickle files in {input_dir}", file=sys.stderr)
        return 1

    index_rows = []
    errors = []

    for pickle_path in tqdm(pickle_files, desc="Exporting to NumPy"):
        try:
            row = export_sample(
                pickle_path,
                output_dir=output_dir,
                with_wavelet=args.with_wavelet,
                overwrite=args.overwrite,
            )
            index_rows.append(row)
        except Exception as exc:
            errors.append((pickle_path.name, str(exc)))

    write_index(index_rows, output_dir / "index.csv")

    exported = sum(1 for r in index_rows if not r.get("skipped"))
    skipped = sum(1 for r in index_rows if r.get("skipped"))
    print(f"Done: {exported} exported, {skipped} skipped, {len(errors)} errors")
    print(f"Output: {output_dir}")
    print(f"Index:  {output_dir / 'index.csv'}")

    if errors:
        print("\nErrors:", file=sys.stderr)
        for name, msg in errors[:10]:
            print(f"  {name}: {msg}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
