"""
bev_generator.py
----------------
Converts Waymo FRONT camera frames into Bird's-Eye View (BEV)
binary occupancy grids using Inverse Perspective Mapping (IPM).

IPM uses a homography matrix derived from camera geometry to
"unfold" the perspective image into a flat top-down view.
Bright regions in the BEV = occupied (road surface, objects).
Dark regions = free space or sky (excluded).

This is the standard camera-to-BEV approach used in:
- Tesla Autopilot (before pure vision BEV)
- BEVDet, BEVFormer, and related work

Usage:
    python bev_generator.py \
        --frames_dir ../data/frames \
        --out_dir    ../data/bev_grids \
        --grid_size  128
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm


# ── IPM homography ────────────────────────────────────────────────────────────

def get_ipm_homography(
    src_h: int = 880,
    src_w: int = 1920,
    bev_h: int = 128,
    bev_w: int = 128,
    grid_size: int = None,
) -> np.ndarray:
    if grid_size is not None:
        bev_h = grid_size
        bev_w = grid_size
    """
    Compute Inverse Perspective Mapping homography matrix.

    Source points: trapezoid in the bottom half of the image
    (the road region visible from a front-facing camera).
    Destination points: rectangle in BEV space.

    These values are tuned for Waymo FRONT camera geometry.
    """
    # Source points — road trapezoid in image space
    # Bottom-left, bottom-right, top-right, top-left
    src = np.float32([
        [src_w * 0.10, src_h * 0.98],   # bottom-left
        [src_w * 0.90, src_h * 0.98],   # bottom-right
        [src_w * 0.65, src_h * 0.55],   # top-right (vanishing)
        [src_w * 0.35, src_h * 0.55],   # top-left  (vanishing)
    ])

    # Destination points — full BEV rectangle
    dst = np.float32([
        [0,       bev_h],   # bottom-left
        [bev_w,   bev_h],   # bottom-right
        [bev_w,   0     ],   # top-right
        [0,       0     ],   # top-left
    ])

    H, _ = cv2.findHomography(src, dst)
    return H


def frame_to_bev(
    img: np.ndarray,
    H: np.ndarray,
    grid_size: int = 128,
    threshold: int = 30,
) -> np.ndarray:
    """
    Apply IPM homography to a BGR image and produce a binary BEV grid.

    Steps:
    1. Warp image to top-down view using H
    2. Convert to grayscale
    3. Apply edge detection (Canny) to find object boundaries
    4. Threshold to binary occupancy grid

    Returns binary np.ndarray [grid_size, grid_size] uint8 (0 or 1).
    """
    # Warp to BEV
    bev = cv2.warpPerspective(img, H, (grid_size, grid_size))

    # Convert to grayscale
    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)

    # Canny edges — detect road markings, vehicles, boundaries
    edges = cv2.Canny(gray, threshold, threshold * 3)

    # Dilate slightly to connect nearby edge pixels
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Binary grid: 1 = occupied/edge, 0 = free space
    binary = (dilated > 0).astype(np.uint8)

    return binary


def process_frames(
    frames_dir: str,
    out_dir: str,
    grid_size: int = 128,
    max_frames: int = 200,
) -> list[str]:
    """
    Process all PNG frames in frames_dir → binary BEV grids saved as NPY.
    Returns list of saved .npy file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(Path(frames_dir).glob("*.png"))[:max_frames]
    if not files:
        raise FileNotFoundError(f"No PNG files found in {frames_dir}")

    print(f"Processing {len(files)} frames → BEV grids ({grid_size}×{grid_size})")

    # Pre-compute homography (same for all frames, same camera)
    H = get_ipm_homography(grid_size=grid_size)

    saved = []
    for f in tqdm(files, desc="Frame → BEV"):
        img = cv2.imread(str(f))
        if img is None:
            continue

        bev = frame_to_bev(img, H, grid_size=grid_size)

        out_path = os.path.join(out_dir, f.stem + ".npy")
        np.save(out_path, bev)
        saved.append(out_path)

    print(f"✓ Saved {len(saved)} BEV grids to {out_dir}")
    return saved


def visualise_bev_sample(
    frames_dir: str,
    out_path: str,
    grid_size: int = 128,
    n_samples: int = 4,
) -> None:
    """
    Save a side-by-side visualisation: original frame | BEV grid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(Path(frames_dir).glob("*.png"))[:n_samples]
    H = get_ipm_homography(grid_size=grid_size)

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples))
    fig.patch.set_facecolor("#111111")

    for i, f in enumerate(files):
        img_bgr = cv2.imread(str(f))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bev     = frame_to_bev(img_bgr, H, grid_size=grid_size)

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Input frame: {f.name[:30]}", color="white", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(bev, cmap="hot", vmin=0, vmax=1)
        axes[i, 1].set_title("BEV occupancy grid (IPM)", color="white", fontsize=10)
        axes[i, 1].axis("off")

    plt.suptitle(
        "Camera → Bird's-Eye View via Inverse Perspective Mapping\n"
        "White = occupied / edge detected   Black = free space",
        color="white", fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"✓ BEV sample visualisation → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Waymo camera frames to BEV occupancy grids"
    )
    parser.add_argument("--frames_dir", required=True,
                        help="Folder containing PNG camera frames")
    parser.add_argument("--out_dir", required=True,
                        help="Output folder for .npy BEV grids")
    parser.add_argument("--grid_size", type=int, default=128,
                        help="BEV grid resolution (default: 128×128)")
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--visualise", action="store_true",
                        help="Save sample visualisation PNG")
    args = parser.parse_args()

    process_frames(args.frames_dir, args.out_dir, args.grid_size, args.max_frames)

    if args.visualise:
        visualise_bev_sample(
            args.frames_dir,
            os.path.join(args.out_dir, "bev_sample.png"),
            args.grid_size
        )
