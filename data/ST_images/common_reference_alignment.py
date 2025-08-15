import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator

def load_tps(moving, target):
    path = f"landmarks/landmarks_{moving}moving_{target}target.csv"
    df = pd.read_csv(path, header=None)

    moving_pts = df[[2, 3]].values.astype(np.float32)
    fixed_pts = df[[4, 5]].values.astype(np.float32)

    moving_pts = moving_pts.reshape(1, -1, 2)
    fixed_pts = fixed_pts.reshape(1, -1, 2)

    tps_img = cv2.createThinPlateSplineShapeTransformer()
    tps_spots = cv2.createThinPlateSplineShapeTransformer()
    matches = [cv2.DMatch(i, i, 0) for i in range(len(df))]
    tps_img.estimateTransformation(fixed_pts, moving_pts, matches)
    tps_spots.estimateTransformation(moving_pts, fixed_pts, matches)

    return tps_img, tps_spots

def apply_tps_to_coords(tps, x, y):
    pts = np.stack([y, x], axis=1).astype(np.float32).reshape(1, -1, 2)
    _, warped = tps.applyTransformation(pts)
    warped = warped.reshape(-1, 2)
    return warped[:, 1], warped[:, 0]  # return x, y

    # x, y = x.astype('float32'), y.astype('float32')
    # newxy = tps.applyTransformation(np.dstack((x.ravel(), y.ravel())))[1]
    # newxy = newxy.reshape([x.size, 2])
    #
    # map_x = newxy[..., 0].astype(np.float32)
    # map_y = newxy[..., 1].astype(np.float32)
    # return map_x, map_y

def transform_coords(spots_x, spots_y, map_x, map_y):
    h, w = map_x.shape
    grid_y = np.arange(h)
    grid_x = np.arange(w)

    interp_x = RegularGridInterpolator((grid_y, grid_x), map_x, bounds_error=False, fill_value=None)
    interp_y = RegularGridInterpolator((grid_y, grid_x), map_y, bounds_error=False, fill_value=None)

    coords = np.stack([spots_y, spots_x], axis=-1)  # shape (N, 2), in (row, col) = (y, x)

    transformed_x = interp_x(coords)
    transformed_y = interp_y(coords)
    return transformed_x, transformed_y

def apply_tps_to_image(tps, image):
    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.astype('float32'), y.astype('float32')
    newxy = tps.applyTransformation(np.dstack((x.ravel(), y.ravel())))[1]
    newxy = newxy.reshape([h, w, 2])

    map_x = newxy[..., 0].astype(np.float32)
    map_y = newxy[..., 1].astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), map_x, map_y

def align_to_reference(slides, reference):
    os.makedirs("aligned_spots", exist_ok=True)
    os.makedirs(f"ref_{reference}_warped_images", exist_ok=True)

    # Preload all landmark-based TPS warps
    img_warps = {}
    spot_warps = {}
    for i in range(len(slides)-1):
        src, tgt = slides[i], slides[i+1]
        img_warps[(src, tgt)], spot_warps[(src, tgt)] = load_tps(src, tgt)

    # Build warp chains from each slide to reference
    for i, slide in enumerate(slides):
        print(f"Processing {slide} → {reference}")
        # Load image and spots
        image = cv2.imread(f"downscaled_images/20201103_ST_HT206B1-S1Fc1{slide}_10p.tif", cv2.IMREAD_UNCHANGED)
        spot_df = pd.read_csv(f"spot_coordinates/{slide}_coordinates.csv", header=None)
        spot_x = spot_df.iloc[:, -1].values.astype(np.float32) / 10
        spot_y = spot_df.iloc[:, -2].values.astype(np.float32) / 10

        # Determine which warps to apply
        if slide == reference:
            aligned_image = image
            aligned_x, aligned_y = spot_x, spot_y
        else:
            # Chain warps forward to reference
            aligned_image = image.copy()
            aligned_x, aligned_y = spot_x.copy(), spot_y.copy()
            path = slides[i:]
            for j in range(len(path) - 1):
                img_warp = img_warps[(path[j], path[j+1])]
                spot_warp = spot_warps[(path[j], path[j+1])]
                aligned_image, map_x, map_y = apply_tps_to_image(img_warp, aligned_image)
                # aligned_x, aligned_y = apply_tps_to_coords(spot_warp, aligned_x, aligned_y)
                _, map_x, map_y = apply_tps_to_image(spot_warp, aligned_image)
                aligned_x, aligned_y = transform_coords(aligned_x, aligned_y, map_x, map_y)


        # Save aligned outputs
        aligned_name = f"{slide}_aligned_to_{reference}"
        spot_df.iloc[:, -1] = aligned_x
        spot_df.iloc[:, -2] = aligned_y
        spot_df.to_csv(f"aligned_spots/{aligned_name}.csv", index=False, header=False)
        cv2.imwrite(f"ref_{reference}_warped_images/{aligned_name}.tif", aligned_image)

        print(f"→ Saved: warped_images/{aligned_name}.tif and aligned_spots/{aligned_name}.csv")


def plot_aligned_spots(slides, reference="U5", spot_range=np.arange(2000, 4000)):
    n = len(slides)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))

    # Flatten axes array for easy indexing
    axes = axes.flatten() if n > 1 else [axes]

    for i, slide in enumerate(slides):
        # Load aligned image
        image_path = f"ref_{reference}_warped_images/{slide}_aligned_to_{reference}.tif"
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[..., ::-1]  # Convert BGR to RGB

        # Load aligned spot coordinates
        # coord_path = f"aligned_spots/{slide}_aligned_to_{reference}.csv"
        coord_path = f"aligned_spots/{slide}_aligned_to_{reference}.csv"
        df = pd.read_csv(coord_path, header=None)
        df = df[df[1] == 1]
        x = df.iloc[..., -1].values.astype(float)
        y = df.iloc[..., -2].values.astype(float)

        # Plot
        ax = axes[i]
        ax.imshow(img)
        ax.scatter(x, y, s=10, c='red', edgecolor='k', alpha=0.3)
        ax.set_title(f"{slide} (aligned to {reference})")
        ax.axis("off")

    # Hide unused axes
    for j in range(len(slides), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

slides = ["U2", "U3", "U4", "U5"]
# slides = ["U3", "U4"]
align_to_reference(slides, reference="U5")
plot_aligned_spots(slides, reference="U5", spot_range=np.arange(2000, 4000))
