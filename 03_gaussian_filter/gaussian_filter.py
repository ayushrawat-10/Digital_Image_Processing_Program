import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hsi(rgb_image):
    if rgb_image.ndim == 2:
        I = rgb_image.astype(np.float64) / 255.0 if np.issubdtype(rgb_image.dtype, np.integer) else rgb_image.astype(np.float64)
        return np.stack([np.zeros_like(I), np.zeros_like(I), np.clip(I, 0.0, 1.0)], axis=-1)
    if rgb_image.ndim == 3 and rgb_image.shape[2] == 4:
        rgb_image = rgb_image[:, :, :3]
    rgb = rgb_image.astype(np.float64) / 255.0 if np.issubdtype(rgb_image.dtype, np.integer) else rgb_image.astype(np.float64)
    if rgb.max() > 1.0: rgb /= 255.0
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    I = (R + G + B) / 3.0
    sum_rgb = R + G + B
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1.0 - (3.0 / (sum_rgb + 1e-7)) * min_rgb
    S[sum_rgb == 0] = 0.0
    S = np.clip(S, 0.0, 1.0)
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B))
    theta = np.arccos(np.clip(num / (den + 1e-7), -1.0, 1.0))
    H = np.degrees(theta)
    H[B > G] = 360.0 - H[B > G]
    H[den < 1e-7] = 0.0
    H[S < 1e-7] = 0.0
    return np.stack([H / 360.0, S, I], axis=-1)

def hsi_to_rgb(hsi_image):
    H = (hsi_image[:, :, 0] * 360.0) % 360.0
    S, I = hsi_image[:, :, 1], hsi_image[:, :, 2]
    R, G, B = np.zeros_like(I), np.zeros_like(I), np.zeros_like(I)
    rad = lambda deg: deg * (np.pi / 180.0)
    
    s1 = (H >= 0.0) & (H < 120.0)
    h1 = rad(H[s1])
    B[s1] = I[s1] * (1.0 - S[s1])
    R[s1] = I[s1] * (1.0 + (S[s1] * np.cos(h1)) / (np.cos(rad(60.0) - h1) + 1e-7))
    G[s1] = 3.0 * I[s1] - (R[s1] + B[s1])

    s2 = (H >= 120.0) & (H < 240.0)
    h2 = rad(H[s2] - 120.0)
    R[s2] = I[s2] * (1.0 - S[s2])
    G[s2] = I[s2] * (1.0 + (S[s2] * np.cos(h2)) / (np.cos(rad(60.0) - h2) + 1e-7))
    B[s2] = 3.0 * I[s2] - (R[s2] + G[s2])

    s3 = (H >= 240.0) & (H < 360.0)
    h3 = rad(H[s3] - 240.0)
    G[s3] = I[s3] * (1.0 - S[s3])
    B[s3] = I[s3] * (1.0 + (S[s3] * np.cos(h3)) / (np.cos(rad(60.0) - h3) + 1e-7))
    R[s3] = 3.0 * I[s3] - (G[s3] + B[s3])

    return np.round(np.clip(np.stack([R, G, B], axis=-1), 0.0, 1.0) * 255.0).astype(np.uint8)

def load_image(filepath=None, max_dim=400):
    if filepath is None:
        for p in ['test_img1.jpg', '../test_img1.jpg', 'sample_image.jpg', '../sample_image.jpg']:
            if os.path.exists(p):
                filepath = p
                break
    raw = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if raw.ndim == 2: rgb = np.stack([raw, raw, raw], axis=-1)
    elif raw.ndim == 3 and raw.shape[2] == 4: rgb = raw[:, :, :3][:, :, ::-1].copy()
    elif raw.ndim == 3 and raw.shape[2] == 3: rgb = raw[:, :, ::-1].copy()
    else: rgb = raw.copy()
    H, W, _ = rgb.shape
    if max(H, W) > max_dim:
        scale = max_dim / float(max(H, W))
        rgb = cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    return rgb

def pad_image(image, pad_h, pad_w):
    H, W = image.shape
    padded = np.zeros((H + 2 * pad_h, W + 2 * pad_w), dtype=np.float64)
    padded[pad_h:pad_h + H, pad_w:pad_w + W] = image
    for r in range(pad_h):
        padded[r, pad_w:pad_w + W] = image[0, :]
        padded[pad_h + H + r, pad_w:pad_w + W] = image[H - 1, :]
    for c in range(pad_w):
        padded[:, c] = padded[:, pad_w]
        padded[:, pad_w + W + c] = padded[:, pad_w + W - 1]
    return padded

def convolve2d_scratch(image, kernel):
    H, W = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = pad_image(image, pad_h, pad_w)
    out = np.zeros((H, W), dtype=np.float64)
    for r in range(H):
        for c in range(W):
            out[r, c] = np.sum(padded[r:r + kh, c:c + kw] * kernel)
    return out

def save_and_show(fig, filename="output.png"):
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

def generate_gaussian_kernel(sigma=1.0):
    kernel = np.zeros((3, 3), dtype=np.float64)
    for i, x in enumerate([-1, 0, 1]):
        for j, y in enumerate([-1, 0, 1]):
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)

def apply_gaussian_filter(I, sigma=1.0):
    kernel = generate_gaussian_kernel(sigma)
    return np.clip(convolve2d_scratch(I, kernel), 0.0, 1.0)

if __name__ == "__main__":
    rgb = load_image()
    hsi = rgb_to_hsi(rgb)
    I_filt = apply_gaussian_filter(hsi[:, :, 2], sigma=1.0)
    hsi[:, :, 2] = I_filt
    rgb_filt = hsi_to_rgb(hsi)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Gaussian Smoothing Filter (3x3 Kernel, sigma=1.0)", fontsize=14, fontweight='bold')
    axes[0].imshow(rgb); axes[0].set_title("Original RGB"); axes[0].axis('off')
    axes[1].imshow(I_filt, cmap='gray'); axes[1].set_title("Filtered Intensity"); axes[1].axis('off')
    axes[2].imshow(rgb_filt); axes[2].set_title("Reconstructed RGB"); axes[2].axis('off')
    save_and_show(fig, os.path.join(os.path.dirname(__file__), "output.png") if __file__ else "output.png")
