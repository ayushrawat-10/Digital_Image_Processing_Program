"""
=============================================================================
Module: spatial_utils.py
Description: Helper Utilities for Manual Spatial Convolution & Padding
Course: Digital Image Processing (DIP)
=============================================================================
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import hsi_conversion


def pad_image(image, pad_h, pad_w, mode='replicate'):
    """
    Manually pad an image using replicate border clamping.
    Supports both 2D single-channel (H, W) and 3D multi-channel (H, W, C).
    """
    if image.ndim == 3:
        H, W, C = image.shape
        channels = [pad_image(image[:, :, c], pad_h, pad_w, mode=mode) for c in range(C)]
        return np.stack(channels, axis=-1)

    H, W = image.shape
    padded = np.zeros((H + 2 * pad_h, W + 2 * pad_w), dtype=np.float64)

    # Insert original image into the center
    padded[pad_h:pad_h + H, pad_w:pad_w + W] = image

    if mode == 'replicate':
        # Replicate top and bottom borders
        for r in range(pad_h):
            padded[r, pad_w:pad_w + W] = image[0, :]
            padded[pad_h + H + r, pad_w:pad_w + W] = image[H - 1, :]

        # Replicate left and right borders (including corners)
        for c in range(pad_w):
            padded[:, c] = padded[:, pad_w]
            padded[:, pad_w + W + c] = padded[:, pad_w + W - 1]

    return padded


def convolve2d_scratch(image, kernel, mode='replicate'):
    """
    Perform 2D spatial convolution manually using nested loops.
    Supports 2D (H, W) and 3D (H, W, C).
    """
    if image.ndim == 3:
        H, W, C = image.shape
        channels = [convolve2d_scratch(image[:, :, c], kernel, mode=mode) for c in range(C)]
        return np.stack(channels, axis=-1)

    H, W = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = pad_image(image, pad_h, pad_w, mode=mode)
    output = np.zeros((H, W), dtype=np.float64)

    for r in range(H):
        for c in range(W):
            window = padded[r:r + kh, c:c + kw]
            output[r, c] = np.sum(window * kernel)

    return output


def load_test_image(filepath=None, max_dim=400):
    """
    Helper to load test image and convert to RGB format.
    Prioritizes 'test_img1.jpg' if present, then 'sample_image.jpg'.
    """
    if filepath is None:
        if os.path.exists("test_img1.jpg"):
            filepath = "test_img1.jpg"
        elif os.path.exists("sample_image.jpg"):
            filepath = "sample_image.jpg"
        else:
            filepath = "sample_image.jpg"
            import create_sample_image
            create_sample_image.create_sample_image(filepath)
    elif not os.path.exists(filepath):
        if os.path.exists("test_img1.jpg"):
            filepath = "test_img1.jpg"
        else:
            filepath = "sample_image.jpg"
            import create_sample_image
            create_sample_image.create_sample_image(filepath)

    raw = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if raw.ndim == 2:
        rgb = np.stack([raw, raw, raw], axis=-1)
    elif raw.ndim == 3 and raw.shape[2] == 4:
        rgb = raw[:, :, :3][:, :, ::-1].copy()
    elif raw.ndim == 3 and raw.shape[2] == 3:
        rgb = raw[:, :, ::-1].copy()
    else:
        rgb = raw.copy()

    H, W, _ = rgb.shape
    if max(H, W) > max_dim:
        scale = max_dim / float(max(H, W))
        new_W = int(W * scale)
        new_H = int(H * scale)
        rgb = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_AREA)

    return rgb


def plot_and_save_result(rgb_original, filtered_intensity, rgb_filtered,
                         title, filename, output_dir="outputs"):
    """
    Standard 3-panel display: Original RGB, Filtered Intensity, Reconstructed RGB.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=15, fontweight='bold')

    axes[0].imshow(rgb_original)
    axes[0].set_title("1. Original RGB Image", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(filtered_intensity, cmap='gray', vmin=0.0, vmax=1.0)
    axes[1].set_title("2. Filtered Intensity (I)", fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(rgb_filtered)
    axes[2].set_title("3. Reconstructed Filtered RGB", fontsize=12)
    axes[2].axis('off')

    filepath = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"-> Saved: {filepath}")
    plt.show()
