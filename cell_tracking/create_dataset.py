# save_cell_masks.py
import os
from pathlib import Path
import numpy as np
import imageio.v2 as imageio

from typing import List, Optional

def save_cell_masks(
    mask_dir: str,
    cell_label: int,
    out_dir: str,
    frames: Optional[List[int]] = None,
    save_npy: bool = True,
    save_tif: bool = False,
    save_stack: bool = True,
) -> dict:
    """
    Extract and save the binary mask (2D matrix) for a given cell at every time stamp.

    Parameters
    ----------
    mask_dir : str
        Directory with per-frame labeled segmentation masks (e.g., .../01_ST/SEG or .../01_GT/SEG).
        Files are expected to be image stacks of *integer labels* in the CTC format.
    cell_label : int
        The integer label of the cell to extract.
    out_dir : str
        Directory where outputs will be saved. Created if missing.
    frames : list[int] | None
        Specific frame indices to process (0-based). If None, process all frames found.
    save_npy : bool
        Save each frame as `mask_cell{L}_frame{idx:03d}.npy` (boolean H×W array).
    save_tif : bool
        Save each frame as `mask_cell{L}_frame{idx:03d}.tif` (0/255 uint8 image).
    save_stack : bool
        Save a single stacked `.npy` volume of shape (T, H, W) with boolean masks.

    Returns
    -------
    dict
        Summary with keys: num_frames, stack_path (if saved), out_files (list of per-frame files).
    """
    mask_dir = Path(mask_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect and sort frame files (CTC uses strict naming / ordering; sorting preserves time order).
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.tif', '.tiff', '.png'))])
    if not mask_files:
        raise FileNotFoundError(f"No mask images found in: {mask_dir}")

    # Determine which frames to process
    all_idx = list(range(len(mask_files)))
    frame_idx = all_idx if frames is None else frames

    saved_files = []
    stack = []

    for idx in frame_idx:
        fpath = mask_dir / mask_files[idx]
        lab = imageio.imread(fpath)  # integer-labeled mask (CTC format)
        cell_mask = (lab == cell_label)  # boolean H×W

        # Save per-frame outputs
        if save_npy:
            out_npy = out_dir / f"mask_cell{cell_label}_frame{idx:03d}.npy"
            np.save(out_npy, cell_mask)
            saved_files.append(str(out_npy))

        if save_tif:
            out_tif = out_dir / f"mask_cell{cell_label}_frame{idx:03d}.tif"
            # store as 0/255 for easy viewing
            imageio.imwrite(out_tif, (cell_mask.astype(np.uint8) * 255))
            saved_files.append(str(out_tif))

        if save_stack:
            stack.append(cell_mask)

    stack_path = None
    if save_stack and stack:
        stack = np.stack(stack, axis=0)  # (T, H, W) boolean
        stack_path = out_dir / f"mask_cell{cell_label}_stack.npy"
        np.save(stack_path, stack)

    return {
        "num_frames": len(frame_idx),
        "stack_path": str(stack_path) if stack_path else None,
        "out_files": saved_files,
    }

if __name__ == "__main__":
    # Example using your paths from data_animation.py
    MASK_DIR = "/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/data/PhC-C2DH-U373/01_ST/SEG"
    CELL_LABEL = 4
    OUT_DIR = "/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/exports/Cell4_masks"

    summary = save_cell_masks(
        mask_dir=MASK_DIR,
        cell_label=CELL_LABEL,
        out_dir=OUT_DIR,
        frames=None,         # or e.g. [0, 1, 2, 3]
        save_npy=True,
        save_tif=False,
        save_stack=True,
    )
    print(summary)
