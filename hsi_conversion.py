"""
=============================================================================
Module: hsi_conversion.py
Description: Manual RGB to HSI and HSI to RGB Color Space Conversions
Course: Digital Image Processing (DIP)
=============================================================================

This module implements color space transformation between RGB (Red, Green, Blue)
and HSI (Hue, Saturation, Intensity) from scratch using NumPy.

Universal Image Type Support:
- 3-Channel RGB Color Images (uint8 or float)
- 4-Channel RGBA Images (Alpha channel is gracefully stripped)
- 2-Channel / Single-Channel Grayscale or Black & White Images (H=0, S=0, I=gray)
- Binary (1-bit / Boolean / Thresholded) Black & White Images
=============================================================================
"""

import numpy as np


def rgb_to_hsi(rgb_image):
    """
    Manually convert an RGB (or Grayscale / B&W) image to HSI color space.

    Parameters:
        rgb_image (np.ndarray): Input image. Can be:
            - 3D array (H, W, 3): Standard RGB image
            - 3D array (H, W, 4): RGBA image (alpha will be dropped)
            - 3D array (H, W, 1): Single channel 3D array
            - 2D array (H, W): Grayscale or Black & White image

    Returns:
        hsi_image (np.ndarray): HSI image of shape (H, W, 3), float64 with
                                H, S, and I components all normalized to [0, 1].
    """
    # -------------------------------------------------------------
    # Case 1: Handle Grayscale / Black & White 2D or (H, W, 1) images
    # -------------------------------------------------------------
    if rgb_image.ndim == 2 or (rgb_image.ndim == 3 and rgb_image.shape[2] == 1):
        if rgb_image.ndim == 3:
            gray = rgb_image[:, :, 0]
        else:
            gray = rgb_image

        # Normalize to float in [0.0, 1.0]
        if gray.dtype == bool:
            I = gray.astype(np.float64)
        elif np.issubdtype(gray.dtype, np.integer):
            I = gray.astype(np.float64) / 255.0
        else:
            I = gray.astype(np.float64).copy()
            if I.max() > 1.0:
                I /= 255.0

        I = np.clip(I, 0.0, 1.0)
        # In black-and-white / grayscale, R = G = B:
        # Saturation is strictly 0 (no color purity), Hue is conventionally 0
        H = np.zeros_like(I)
        S = np.zeros_like(I)
        return np.stack([H, S, I], axis=-1)

    # -------------------------------------------------------------
    # Case 2: Handle RGBA (4 channels) by stripping Alpha
    # -------------------------------------------------------------
    if rgb_image.ndim == 3 and rgb_image.shape[2] == 4:
        rgb_image = rgb_image[:, :, :3]

    # -------------------------------------------------------------
    # Case 3: Standard 3-Channel RGB Image
    # -------------------------------------------------------------
    if rgb_image.dtype == bool:
        rgb = rgb_image.astype(np.float64)
    elif np.issubdtype(rgb_image.dtype, np.integer):
        rgb = rgb_image.astype(np.float64) / 255.0
    else:
        rgb = rgb_image.astype(np.float64).copy()
        if rgb.max() > 1.0:
            rgb /= 255.0

    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    # Compute Intensity component I
    I = (R + G + B) / 3.0

    # Compute Saturation component S
    sum_rgb = R + G + B
    min_rgb = np.minimum(np.minimum(R, G), B)

    epsilon = 1e-7
    S = 1.0 - (3.0 / (sum_rgb + epsilon)) * min_rgb
    S[sum_rgb == 0] = 0.0
    S = np.clip(S, 0.0, 1.0)

    # Compute Hue component H
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))

    cos_theta = np.clip(num / (den + epsilon), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    theta_deg = np.degrees(theta)

    H_deg = np.zeros_like(I)
    mask_b_le_g = (B <= G)
    H_deg[mask_b_le_g] = theta_deg[mask_b_le_g]
    H_deg[~mask_b_le_g] = 360.0 - theta_deg[~mask_b_le_g]

    # If achromatic / grayscale pixel (R=G=B or S=0), Hue is 0
    H_deg[den < epsilon] = 0.0
    H_deg[S < epsilon] = 0.0

    # Normalize Hue to [0, 1]
    H = H_deg / 360.0

    return np.stack([H, S, I], axis=-1)


def hsi_to_rgb(hsi_image):
    """
    Manually convert an HSI image back to RGB color space.

    Parameters:
        hsi_image (np.ndarray): Input HSI image, float with H, S, I in [0, 1].

    Returns:
        rgb_image (np.ndarray): Output RGB image, uint8 in range [0, 255].
    """
    H = hsi_image[:, :, 0].copy()
    S = hsi_image[:, :, 1].copy()
    I = hsi_image[:, :, 2].copy()

    H_deg = (H * 360.0) % 360.0

    R = np.zeros_like(I)
    G = np.zeros_like(I)
    B = np.zeros_like(I)

    def deg2rad(deg):
        return deg * (np.pi / 180.0)

    # Sector 1: Red-Green sector (0 <= H < 120 degrees)
    sector1 = (H_deg >= 0.0) & (H_deg < 120.0)
    h_rad1 = deg2rad(H_deg[sector1])
    s1 = S[sector1]
    i1 = I[sector1]

    b1 = i1 * (1.0 - s1)
    r1 = i1 * (1.0 + (s1 * np.cos(h_rad1)) / (np.cos(deg2rad(60.0) - h_rad1) + 1e-7))
    g1 = 3.0 * i1 - (r1 + b1)

    R[sector1] = r1
    G[sector1] = g1
    B[sector1] = b1

    # Sector 2: Green-Blue sector (120 <= H < 240 degrees)
    sector2 = (H_deg >= 120.0) & (H_deg < 240.0)
    h_rad2 = deg2rad(H_deg[sector2] - 120.0)
    s2 = S[sector2]
    i2 = I[sector2]

    r2 = i2 * (1.0 - s2)
    g2 = i2 * (1.0 + (s2 * np.cos(h_rad2)) / (np.cos(deg2rad(60.0) - h_rad2) + 1e-7))
    b2 = 3.0 * i2 - (r2 + g2)

    R[sector2] = r2
    G[sector2] = g2
    B[sector2] = b2

    # Sector 3: Blue-Red sector (240 <= H < 360 degrees)
    sector3 = (H_deg >= 240.0) & (H_deg < 360.0)
    h_rad3 = deg2rad(H_deg[sector3] - 240.0)
    s3 = S[sector3]
    i3 = I[sector3]

    g3 = i3 * (1.0 - s3)
    b3 = i3 * (1.0 + (s3 * np.cos(h_rad3)) / (np.cos(deg2rad(60.0) - h_rad3) + 1e-7))
    r3 = 3.0 * i3 - (g3 + b3)

    R[sector3] = r3
    G[sector3] = g3
    B[sector3] = b3

    # Stack channels and clip to valid range [0, 1]
    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)

    # Convert to standard 8-bit unsigned integer [0, 255]
    rgb_uint8 = np.round(rgb * 255.0).astype(np.uint8)
    return rgb_uint8
