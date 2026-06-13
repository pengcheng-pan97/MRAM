import argparse, os, json, pickle, re, numpy as np
from pathlib import Path
from typing import List

# ---------------- I‑DT helper ----------------

def extract_fixations(
    gaze_xy: np.ndarray,
    spatial_thresh: float = 15.0,
    min_pts: int = 2,
    target_n: int = 7,
) -> np.ndarray:
    """Return *target_n* fixation centroids (x,y) in temporal order."""
    clusters, cur = [], []
    for p in gaze_xy:  # p : (x,y)
        if not cur:
            cur.append(p)
            continue
        if np.linalg.norm(p - np.mean(cur, axis=0)) < spatial_thresh:
            cur.append(p)
        else:
            if len(cur) >= min_pts:
                clusters.append(np.mean(cur, axis=0))
            cur = [p]
    if len(cur) >= min_pts:
        clusters.append(np.mean(cur, axis=0))

    if len(clusters) >= target_n:
        clusters = clusters[:target_n]
    else:
        clusters += [clusters[-1]] * (target_n - len(clusters))
    return np.stack(clusters[:target_n], axis=0)  # (target_n,2)

# --------------- file I/O helpers ------------

def parse_gaze_txt(path: Path) -> np.ndarray:
    """Load (x,y) pairs from a ( line, e.g. "(158.128, 23.949)"."""
    coords: List[List[float]] = []
    patt = re.compile(r"\(?\s*([\d.]+)\s*,\s*([\d.]+)\s*\)?")
    with path.open() as f:
        for line in f:
            m = patt.search(line)
            if m:
                coords.append([float(m.group(1)), float(m.group(2))])
    if not coords:
        raise ValueError(f"No coordinates parsed from {path}")
    return np.asarray(coords, dtype=np.float32)


def save_fixations(path_out: Path, fix: np.ndarray, fmt: str):
    if fmt == ".json":
        with path_out.open("w") as f:
            json.dump({"fixations": fix.tolist()}, f)
    else:  # ".p"
        with path_out.open("wb") as f:
            pickle.dump(fix, f)

# --------------- main ------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre‑cluster Gaze‑CIFAR‑10 gaze streams into 12 fixations.")
    parser.add_argument("root", type=str, default="../Gaze-CIFAR-10-0/test data", nargs="?",
                        help="Root folder that contains class subfolders 0 … 9 (default: %(default)s)")
    parser.add_argument("--ext", choices=[".json", ".p"], default=".json",
                        help="Output file extension / format (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if output exists")
    args = parser.parse_args()

    ROOT = Path(args.root)
    assert ROOT.exists(), f"Folder not found: {ROOT}"

    txt_files = sorted(list(ROOT.glob("*/p*.txt")))  # recurse into class folders
    print(f"Found {len(txt_files)} gaze files … processing …")

    for txt_path in txt_files:
        out_path = txt_path.with_suffix(args.ext)
        if out_path.exists() and not args.overwrite:
            continue
        try:
            raw_xy = parse_gaze_txt(txt_path)
            fix12 = extract_fixations(raw_xy)
            save_fixations(out_path, fix12, args.ext)
        except Exception as e:
            print(f"[warn] skipping {txt_path} :: {e}")

    print("Done.")


if __name__ == "__main__":
    main()