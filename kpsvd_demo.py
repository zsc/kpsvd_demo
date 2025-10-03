#!/usr/bin/env python3
"""
KPSVD Demo: Kronecker Product SVD with k-rank approximation and noise visualization
Uses Van Loan-Pitsianis method for nearest Kronecker sum approximation
"""

import numpy as np
from PIL import Image
import base64
from io import BytesIO
import os


def image_to_grayscale_matrix(image_path):
    """Convert image to grayscale matrix, cropping dimensions to multiples of 32"""
    img = Image.open(image_path).convert('L')

    # Crop to multiples of 32
    w, h = img.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32

    if new_w > 0 and new_h > 0:
        # Center crop
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h))
        print(f"Cropped from {w}x{h} to {new_w}x{new_h}")

    return np.array(img, dtype=float)


def kpsvd(M, k):
    """
    Kronecker Product SVD using Van Loan-Pitsianis method

    Args:
        M: Input matrix (m x n)
        k: Rank for truncated SVD

    Returns:
        U, S, Vt: Left factor, singular values, right factor from k-rank approximation
    """
    m, n = M.shape

    # Create rearranged matrix R(M)
    # For Van Loan-Pitsianis, we reshape M into a square-like form
    p = int(np.sqrt(m))
    q = int(np.sqrt(n))

    # Adjust to closest factors
    while m % p != 0 and p > 1:
        p -= 1
    while n % q != 0 and q > 1:
        q -= 1

    r = m // p
    s = n // q

    # Rearrange M into R(M) of size (p*q) x (r*s)
    R = np.zeros((p * q, r * s))

    for i in range(p):
        for j in range(q):
            block = M[i*r:(i+1)*r, j*s:(j+1)*s]
            R[i*q + j, :] = block.flatten()

    # Perform truncated SVD on R(M)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)

    # Truncate to rank k
    k = min(k, len(S))
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    return U_k, S_k, Vt_k, (p, q, r, s)


def reconstruct_from_kpsvd(U_k, S_k, Vt_k, shape_info):
    """
    Reconstruct matrix from KPSVD factors

    Args:
        U_k, S_k, Vt_k: KPSVD factors
        shape_info: (p, q, r, s) dimensions tuple

    Returns:
        Reconstructed matrix
    """
    p, q, r, s = shape_info

    # Reconstruct R(M)
    R_approx = U_k @ np.diag(S_k) @ Vt_k

    # Reverse the rearrangement
    m, n = p * r, q * s
    M_approx = np.zeros((m, n))

    for i in range(p):
        for j in range(q):
            block = R_approx[i*q + j, :].reshape((r, s))
            M_approx[i*r:(i+1)*r, j*s:(j+1)*s] = block

    return M_approx


def add_noise_to_factor(factor, noise_levels):
    """
    Generate series of noisy versions of a factor

    Args:
        factor: Input matrix (U or V)
        noise_levels: List of noise standard deviations

    Returns:
        List of noisy factors
    """
    noisy_factors = []
    for noise_std in noise_levels:
        noise = np.random.normal(0, noise_std, factor.shape)
        noisy_factors.append(factor + noise)
    return noisy_factors


def matrix_to_image(matrix):
    """Convert matrix to PIL Image, clipping to [0, 255]"""
    clipped = np.clip(matrix, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped)


def image_to_base64(img):
    """Convert PIL Image to base64 string for HTML embedding"""
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def generate_html_visualization(original, approximation, left_noise_images, right_noise_images,
                                 k, noise_levels, output_path='kpsvd_visualization.html'):
    """
    Generate HTML file with all visualizations

    Args:
        original: Original grayscale matrix
        approximation: k-rank approximation
        left_noise_images: List of images with left factor noise
        right_noise_images: List of images with right factor noise
        k: Rank used for approximation
        noise_levels: Noise levels used
        output_path: Path to save HTML file
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KPSVD Visualization (k={k})</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2 {{
            color: #333;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .image-container {{
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .image-label {{
            margin-top: 10px;
            font-weight: bold;
            color: #555;
        }}
        .stats {{
            background: #f9f9f9;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>KPSVD Visualization: Van Loan-Pitsianis Method</h1>

    <div class="section">
        <h2>Original and k-rank Approximation (k={k})</h2>
        <div class="image-grid">
            <div class="image-container">
                <img src="{image_to_base64(matrix_to_image(original))}" alt="Original">
                <div class="image-label">Original Image</div>
                <div class="stats">Shape: {original.shape}</div>
            </div>
            <div class="image-container">
                <img src="{image_to_base64(matrix_to_image(approximation))}" alt="Approximation">
                <div class="image-label">k={k} Approximation</div>
                <div class="stats">
                    MSE: {np.mean((original - approximation)**2):.2f}<br>
                    PSNR: {20 * np.log10(255 / np.sqrt(np.mean((original - approximation)**2))):.2f} dB
                </div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Left Factor Noise Series</h2>
        <p>Noise added to U (left factor), noise levels: {noise_levels}</p>
        <div class="image-grid">
"""

    for i, (img, noise_level) in enumerate(zip(left_noise_images, noise_levels)):
        html_content += f"""
            <div class="image-container">
                <img src="{image_to_base64(matrix_to_image(img))}" alt="Left noise {i}">
                <div class="image-label">U noise σ={noise_level}</div>
            </div>
"""

    html_content += """
        </div>
    </div>

    <div class="section">
        <h2>Right Factor Noise Series</h2>
        <p>Noise added to V (right factor), noise levels: """ + str(noise_levels) + """</p>
        <div class="image-grid">
"""

    for i, (img, noise_level) in enumerate(zip(right_noise_images, noise_levels)):
        html_content += f"""
            <div class="image-container">
                <img src="{image_to_base64(matrix_to_image(img))}" alt="Right noise {i}">
                <div class="image-label">V noise σ={noise_level}</div>
            </div>
"""

    html_content += """
        </div>
    </div>

    <div class="section">
        <h2>Method Description</h2>
        <p><strong>KPSVD (Kronecker Product SVD)</strong> using the Van Loan-Pitsianis method:</p>
        <ol>
            <li>Rearrange matrix M into R(M)</li>
            <li>Perform truncated SVD on R(M): R(M) ≈ U<sub>k</sub> Σ<sub>k</sub> V<sub>k</sub><sup>T</sup></li>
            <li>Reconstruct approximation from k-rank factors</li>
            <li>Add Gaussian noise to left (U) and right (V) factors separately</li>
        </ol>
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"HTML visualization saved to: {output_path}")


def main(image_path, k=10, noise_levels=None, output_html='kpsvd_visualization.html'):
    """
    Main function to run KPSVD demo

    Args:
        image_path: Path to input image
        k: Rank for approximation
        noise_levels: List of noise standard deviations (default: [5, 10, 20])
        output_html: Output HTML file path
    """
    if noise_levels is None:
        noise_levels = [5, 10, 20]

    print(f"Loading image: {image_path}")
    M = image_to_grayscale_matrix(image_path)
    print(f"Image shape: {M.shape}")

    print(f"\nRunning KPSVD with k={k}...")
    U_k, S_k, Vt_k, shape_info = kpsvd(M, k)
    print(f"KPSVD factors: U_k {U_k.shape}, S_k {S_k.shape}, Vt_k {Vt_k.shape}")

    print("\nReconstructing k-rank approximation...")
    M_approx = reconstruct_from_kpsvd(U_k, S_k, Vt_k, shape_info)

    print(f"\nGenerating left factor noise series (levels: {noise_levels})...")
    left_noisy_factors = add_noise_to_factor(U_k, noise_levels)
    left_noise_images = []
    for noisy_U in left_noisy_factors:
        img = reconstruct_from_kpsvd(noisy_U, S_k, Vt_k, shape_info)
        left_noise_images.append(img)

    print(f"Generating right factor noise series (levels: {noise_levels})...")
    right_noisy_factors = add_noise_to_factor(Vt_k.T, noise_levels)
    right_noise_images = []
    for noisy_V in right_noisy_factors:
        img = reconstruct_from_kpsvd(U_k, S_k, noisy_V.T, shape_info)
        right_noise_images.append(img)

    print(f"\nGenerating HTML visualization...")
    generate_html_visualization(M, M_approx, left_noise_images, right_noise_images,
                                k, noise_levels, output_html)

    print("\nDone!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kpsvd_demo.py <image_path> [k] [noise_levels...]")
        print("Example: python kpsvd_demo.py image.jpg 10 5 10 20")
        sys.exit(1)

    image_path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    noise_levels = [float(x) for x in sys.argv[3:]] if len(sys.argv) > 3 else [5, 10, 20]

    main(image_path, k, noise_levels)
