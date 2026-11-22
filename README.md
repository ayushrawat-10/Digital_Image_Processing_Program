# Digital Image Processing (DIP) Lab Project

A practical, student-friendly implementation of spatial domain filters, frequency domain filters, and lossless data compression algorithms written from scratch in Python for our B.Tech CSE Digital Image Processing assignment.

---

## What is this Project?

In image processing, applying filters directly to Red, Green, and Blue (RGB) channels usually ruins the original colors. To solve this:
1. We convert the RGB image into the **HSI (Hue, Saturation, Intensity)** color model.
2. We apply our filters **only to the Intensity ($I$) channel** (brightness and structure).
3. The original Hue ($H$) and Saturation ($S$) are kept unchanged.
4. We convert back to RGB, giving us a clean filtered image with perfectly preserved colors!

All algorithms (convolution, padding, FFT masks, Huffman, and Arithmetic coding) are built **from scratch** using basic NumPy math—no ready-made OpenCV/SciPy filter functions.

---

## Project Folder Structure

Each filter and compression technique is organized into its own separate folder containing its **code** and its **sample output**:

```
DIP_Filter/
│
├── 01_mean_filter/                     # Code + output.png
├── 02_weighted_average_filter/         # Code + output.png
├── 03_gaussian_filter/                 # Code + output.png
├── 04_median_filter/                   # Code + output.png
├── 05_laplacian_filter/                # Code + output.png
├── 06_sobel_filter/                    # Code + output.png
├── 07_high_boost_filter/               # Code + output.png
├── 08_ideal_low_pass_filter/           # Code + output.png
├── 09_butterworth_low_pass_filter/     # Code + output.png
├── 10_gaussian_low_pass_filter/        # Code + output.png
├── 11_ideal_high_pass_filter/          # Code + output.png
├── 12_butterworth_high_pass_filter/    # Code + output.png
├── 13_gaussian_high_pass_filter/       # Code + output.png
├── 14_huffman_encoding/                # Code + output.txt
├── 15_arithmetic_encoding/             # Code + output.txt
│
├── outputs/                            # All combined comparison plots
├── main.py                             # Interactive all-in-one menu
├── test_img1.jpg                       # Test input image
├── requirements.txt                    # opencv-python, numpy, matplotlib
└── README.md
```

---

## Quick Summary of Filters

### A. Spatial Domain Filters (Manual 2D Convolution)

1. **Mean / Averaging Filter (`01_mean_filter/`)**:
   - Replaces each pixel with the average of its $3 \times 3$ neighborhood.
   - Mask: $\frac{1}{9} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$

2. **Weighted Average Filter (`02_weighted_average_filter/`)**:
   - Gives highest weight to the center pixel to reduce edge blurring.
   - Mask: $\frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$

3. **Gaussian Filter (`03_gaussian_filter/`)**:
   - Smooths noise based on a 2D Gaussian bell curve ($G(x, y) = \frac{1}{2\pi \sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$).

4. **Median Filter (`04_median_filter/`)**:
   - Non-linear filter. Sorts the 9 neighboring pixels and picks the middle value.
   - Best for completely removing **Salt & Pepper noise** without blurring edges.

5. **Laplacian Sharpening (`05_laplacian_filter/`)**:
   - Second-derivative operator ($\nabla^2 f$) with mask $\begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$.
   - Enhances edges: $\text{Sharpened} = f(x, y) - \nabla^2 f(x, y)$.

6. **Sobel Edge Detection (`06_sobel_filter/`)**:
   - Uses horizontal ($G_x$) and vertical ($G_y$) derivative operators.
   - Computes gradient magnitude: $\sqrt{G_x^2 + G_y^2}$.

7. **High Boost Filter (`07_high_boost_filter/`)**:
   - Amplifies high-frequency edges while keeping overall brightness:
     $$\text{High Boost} = A \cdot \text{Original} - \text{Blurred} \quad (\text{default } A = 1.5)$$

---

### B. Frequency Domain Filters (2D FFT)

All frequency filters compute 2D FFT, shift zero frequency to center, multiply by mask $H(u, v)$, and take inverse FFT:

8. **Ideal Low Pass Filter (ILPF) (`08_ideal_low_pass_filter/`)**:
   - Hard circular cutoff ($D \le D_0$). Produces visible ringing (Gibbs phenomenon).

9. **Butterworth Low Pass Filter (BLPF) (`09_butterworth_low_pass_filter/`)**:
   - Smooth transition: $H(u, v) = \frac{1}{1 + (D/D_0)^{2n}}$. At $n=2$, smooths with no ringing.

10. **Gaussian Low Pass Filter (GLPF) (`10_gaussian_low_pass_filter/`)**:
    - Exponential decay: $H(u, v) = e^{-D^2 / (2 D_0^2)}$. Zero ringing.

11. **Ideal High Pass Filter (IHPF) (`11_ideal_high_pass_filter/`)**:
    - Complement of ILPF ($1 - H_{\text{ILPF}}$). Passes edges, blocks low frequencies.

12. **Butterworth High Pass Filter (BHPF) (`12_butterworth_high_pass_filter/`)**:
    - Complement of BLPF ($1 - H_{\text{BLPF}}$). Smooth high-pass edge filtering.

13. **Gaussian High Pass Filter (GHPF) (`13_gaussian_high_pass_filter/`)**:
    - Complement of GLPF ($1 - H_{\text{GLPF}}$). Clean edge extraction without ringing.

---

### C. Image Compression Techniques

14. **Huffman Encoding (`14_huffman_encoding/`)**:
    - Variable-length prefix code based on symbol frequencies.
    - Frequent symbols get short codes; rare symbols get long codes.
    - Demonstrated on both text (`"BANANA"`) and actual image pixel data (histogram-based encoding).

15. **Arithmetic Encoding (`15_arithmetic_encoding/`)**:
    - Encodes an entire sequence into a single fractional number in $[0.0, 1.0)$ by successively narrowing intervals.
    - Decodes back with 100% exact match.

---

## How to Run

### Option 1: Run any filter folder individually
Go into any folder and run its Python script:
```bash
# Example: Run Mean Filter
cd 01_mean_filter
python mean_filter.py

# Example: Run Sobel Filter
cd ../06_sobel_filter
python sobel_filter.py

# Example: Run Huffman Coding
cd ../14_huffman_encoding
python huffman_encoding.py
```

### Option 2: Run all filters at once from root
```bash
# Automated batch (runs all filters and saves all outputs)
python main.py --all

# Or interactive menu
python main.py
```

---

## Requirements

Install the 3 required libraries:
```bash
pip install -r requirements.txt
```
*(Only `opencv-python`, `numpy`, and `matplotlib`)*
