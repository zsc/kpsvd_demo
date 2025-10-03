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


def kpsvd(M, k, left_scale):
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

    if left_scale is not None:
        p *= left_scale 
        q *= left_scale

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


def downscale_upscale_matrix(matrix, scale_factor):
    """
    Downscale then upscale matrix back to original size

    Args:
        matrix: Input matrix
        scale_factor: Scale factor (e.g., 0.5 for half size)

    Returns:
        Matrix after downscale->upscale operation
    """
    img = matrix_to_image(matrix)
    original_size = img.size

    if scale_factor >= 1.0:
        return matrix

    # Downscale
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    if new_size[0] < 1 or new_size[1] < 1:
        new_size = (1, 1)

    img_down = img.resize(new_size, Image.BILINEAR)

    # Upscale back to original size
    img_up = img_down.resize(original_size, Image.BILINEAR)

    return np.array(img_up, dtype=float)


def downscale_upscale_factor(factor, scale_factor, r, s):
    """
    Downscale then upscale a factor matrix by treating each of its k columns
    as a separate (r, s) image.

    Args:
        factor: Input factor matrix of shape (r*s, k).
        scale_factor: Scale factor for resizing (e.g., 0.5 for half size).
        r: The number of rows for the reshaped image-like blocks.
        s: The number of columns for the reshaped image-like blocks.

    Returns:
        Factor matrix of shape (r*s, k) after downscale->upscale operation.
    """
    # If no scaling is needed, return the original factor
    if scale_factor >= 1.0:
        return factor

    # Get the number of columns (k) from the factor's shape
    # This also handles the case where the factor might be a 1D array (k=1)
    if factor.ndim == 1:
        k = 1
        factor = factor.reshape(-1, 1) # Ensure it's a column vector
    else:
        k = factor.shape[1]

    # Pre-allocate the result matrix for efficiency
    result_matrix = np.zeros_like(factor)
    original_pil_size = (s, r) # PIL's size is (width, height), which corresponds to (s, r)

    # Iterate through each of the k columns
    for i in range(k):
        # 1. Extract the column and reshape it into an (r, s) image matrix
        column_vector = factor[:, i]
        image_matrix = column_vector.reshape((r, s))

        # 2. Convert to a PIL Image for resizing operations
        # Using float32 is good practice for compatibility with PIL's filters
        img = Image.fromarray(image_matrix.astype(np.float32))

        # 3. Downscale the image
        # Calculate the new size, ensuring dimensions are at least 1 pixel
        new_width = max(1, int(img.width * scale_factor))
        new_height = max(1, int(img.height * scale_factor))
        new_size = (new_width, new_height)
        img_down = img.resize(new_size, Image.BILINEAR)

        # 4. Upscale the image back to its original size
        img_up = img_down.resize(original_pil_size, Image.BILINEAR)

        # 5. Convert the processed PIL image back to a NumPy array,
        #    flatten it, and place it into the corresponding column of the result matrix.
        processed_image_matrix = np.array(img_up, dtype=float)
        result_matrix[:, i] = processed_image_matrix.flatten()

    return result_matrix

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


def generate_html_visualization(original, all_k_results, noise_levels, scale_factors, shape_info, output_path='kpsvd_visualization.html'):
    """
    Generate HTML file with all visualizations

    Args:
        original: Original grayscale matrix
        all_k_results: Dict with k values as keys, each containing:
            - approximation: k-rank approximation
            - left_noise_images: List of images with left factor noise
            - right_noise_images: List of images with right factor noise
            - original_scale_images: List of downscale-upscale images of original
            - right_factor_scale_images: List of downscale-upscale images of right factor
            - compression_ratio: Compression ratio
        noise_levels: Noise levels used
        scale_factors: Scale factors used for downscale-upscale
        shape_info: (p, q, r, s) dimensions
        output_path: Path to save HTML file
    """
    p, q, r, s = shape_info
    m, n = original.shape
    k_values = sorted(all_k_results.keys())
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KPSVD Visualization</title>
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
        .controls {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }}
        .slider-container {{
            margin: 15px 0;
        }}
        .slider-container label {{
            font-weight: bold;
            display: block;
            margin-bottom: 5px;
        }}
        .slider-container input[type="range"] {{
            width: 100%;
            max-width: 500px;
        }}
        .slider-value {{
            display: inline-block;
            margin-left: 10px;
            font-weight: bold;
            color: #0066cc;
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
            text-align: left;
        }}
        .k-section {{
            display: none;
        }}
        .k-section.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <h1>KPSVD Visualization: Van Loan-Pitsianis Method</h1>

    <div class="section">
        <h2>Parameters</h2>
        <div class="stats">
            <strong>Original Image:</strong> {m} × {n}<br>
            <strong>KPSVD Dimensions:</strong> p={p}, q={q}, r={r}, s={s}<br>
            <strong>Rearranged Matrix R(M):</strong> {p*q} × {r*s}
        </div>
    </div>

    <div class="section controls">
        <div class="slider-container">
            <label for="k-slider">Rank k: <span class="slider-value" id="k-value">{k_values[0]}</span></label>
            <input type="range" id="k-slider" min="0" max="{len(k_values)-1}" value="0" step="1">
            <div style="margin-top: 5px; color: #666; font-size: 0.9em;">
                Available k values: {', '.join(map(str, k_values))}
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Original Image</h2>
        <div class="image-grid">
            <div class="image-container">
                <img src="{image_to_base64(matrix_to_image(original))}" alt="Original">
                <div class="image-label">Original Image</div>
                <div class="stats">
                    Shape: {original.shape}<br>
                    Total elements: {m * n}
                </div>
            </div>
        </div>
    </div>
"""

    # Generate sections for each k value
    for i, k in enumerate(k_values):
        result = all_k_results[k]
        approx = result['approximation']
        left_imgs = result['left_noise_images']
        right_imgs = result['right_noise_images']
        compression = result['compression_ratio']

        mse = np.mean((original - approx)**2)
        psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')

        active_class = 'active' if i == 0 else ''

        html_content += f"""
    <div class="k-section {active_class}" data-k-index="{i}">
        <div class="section">
            <h2>k={k} Approximation</h2>
            <div class="image-grid">
                <div class="image-container">
                    <img src="{image_to_base64(matrix_to_image(approx))}" alt="Approximation k={k}">
                    <div class="image-label">k={k} Approximation</div>
                    <div class="stats">
                        <strong>Compression Ratio:</strong> {compression:.2f}×<br>
                        <strong>Storage:</strong> {p*q*k + k + r*s*k} elements (vs {m*n} original)<br>
                        <strong>MSE:</strong> {mse:.2f}<br>
                        <strong>PSNR:</strong> {psnr:.2f} dB
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Left Factor Noise Series (k={k})</h2>
            <p>Noise added to U (left factor), noise levels: {noise_levels}</p>
            <div class="image-grid">
"""

        for j, (img, noise_level) in enumerate(zip(left_imgs, noise_levels)):
            img_mse = np.mean((original - img)**2)
            img_psnr = 20 * np.log10(255 / np.sqrt(img_mse)) if img_mse > 0 else float('inf')
            html_content += f"""
                <div class="image-container">
                    <img src="{image_to_base64(matrix_to_image(img))}" alt="Left noise {j}">
                    <div class="image-label">U noise σ={noise_level}</div>
                    <div class="stats">PSNR: {img_psnr:.2f} dB</div>
                </div>
"""

        html_content += """
            </div>
        </div>

        <div class="section">
            <h2>Right Factor Noise Series (k=""" + str(k) + """)</h2>
            <p>Noise added to V (right factor), noise levels: """ + str(noise_levels) + """</p>
            <div class="image-grid">
"""

        for j, (img, noise_level) in enumerate(zip(right_imgs, noise_levels)):
            img_mse = np.mean((original - img)**2)
            img_psnr = 20 * np.log10(255 / np.sqrt(img_mse)) if img_mse > 0 else float('inf')
            html_content += f"""
                <div class="image-container">
                    <img src="{image_to_base64(matrix_to_image(img))}" alt="Right noise {j}">
                    <div class="image-label">V noise σ={noise_level}</div>
                    <div class="stats">PSNR: {img_psnr:.2f} dB</div>
                </div>
"""

        html_content += """
            </div>
        </div>

        <div class="section">
            <h2>Original Image Downscale-Upscale Series (k=""" + str(k) + """)</h2>
            <p>Original image downscaled then upscaled back, scale factors: """ + str(scale_factors) + """</p>
            <div class="image-grid">
"""

        original_scale_imgs = result['original_scale_images']
        for j, (img, scale) in enumerate(zip(original_scale_imgs, scale_factors)):
            img_mse = np.mean((original - img)**2)
            img_psnr = 20 * np.log10(255 / np.sqrt(img_mse)) if img_mse > 0 else float('inf')
            html_content += f"""
                <div class="image-container">
                    <img src="{image_to_base64(matrix_to_image(img))}" alt="Original scale {j}">
                    <div class="image-label">Original scale {scale}×</div>
                    <div class="stats">PSNR: {img_psnr:.2f} dB</div>
                </div>
"""

        html_content += """
            </div>
        </div>

        <div class="section">
            <h2>Right Factor Downscale-Upscale Series (k=""" + str(k) + """)</h2>
            <p>Right factor (V) downscaled then upscaled back, scale factors: """ + str(scale_factors) + """</p>
            <div class="image-grid">
"""

        right_scale_imgs = result['right_factor_scale_images']
        for j, (img, scale) in enumerate(zip(right_scale_imgs, scale_factors)):
            img_mse = np.mean((original - img)**2)
            img_psnr = 20 * np.log10(255 / np.sqrt(img_mse)) if img_mse > 0 else float('inf')
            html_content += f"""
                <div class="image-container">
                    <img src="{image_to_base64(matrix_to_image(img))}" alt="Right factor scale {j}">
                    <div class="image-label">V scale {scale}×</div>
                    <div class="stats">PSNR: {img_psnr:.2f} dB</div>
                </div>
"""

        html_content += """
            </div>
        </div>
    </div>
"""

    html_content += """
    <div class="section">
        <h2>Method Description</h2>
        <p><strong>KPSVD (Kronecker Product SVD)</strong> using the Van Loan-Pitsianis method:</p>
        <ol>
            <li>Rearrange matrix M into R(M)</li>
            <li>Perform truncated SVD on R(M): R(M) ≈ U<sub>k</sub> Σ<sub>k</sub> V<sub>k</sub><sup>T</sup></li>
            <li>Reconstruct approximation from k-rank factors</li>
            <li>Add Gaussian noise to left (U) and right (V) factors separately</li>
        </ol>
        <p><strong>Compression Ratio:</strong> Original size / Compressed size = (m×n) / (p×q×k + k + r×s×k)</p>
    </div>

    <script>
        const slider = document.getElementById('k-slider');
        const kValue = document.getElementById('k-value');
        const kSections = document.querySelectorAll('.k-section');
        const kValues = [""" + ','.join(map(str, k_values)) + """];

        slider.addEventListener('input', function() {
            const index = parseInt(this.value);
            kValue.textContent = kValues[index];

            // Hide all sections
            kSections.forEach(section => section.classList.remove('active'));

            // Show selected section
            const selectedSection = document.querySelector(`[data-k-index="${index}"]`);
            if (selectedSection) {
                selectedSection.classList.add('active');
            }
        });
    </script>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"HTML visualization saved to: {output_path}")


def main(image_path, left_scale=None, k_values=None, noise_levels=None, scale_factors=None, output_html='kpsvd_visualization.html'):
    """
    Main function to run KPSVD demo

    Args:
        image_path: Path to input image
        k_values: List of k ranks to compute (default: [5, 10, 20, 50])
        noise_levels: List of noise standard deviations (default: [5, 10, 20])
        scale_factors: List of scale factors for downscale-upscale (default: [1.0, 0.5, 0.25, 0.125])
        output_html: Output HTML file path
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]
    if noise_levels is None:
        noise_levels = [5, 10, 20]
    if scale_factors is None:
        scale_factors = [1.0, 0.5, 0.25, 0.125]

    print(f"Loading image: {image_path}")
    M = image_to_grayscale_matrix(image_path)
    print(f"Image shape: {M.shape}")
    m, n = M.shape

    # Store results for all k values
    all_k_results = {}
    shape_info = None

    for k in k_values:
        print(f"\nRunning KPSVD with k={k}...")
        U_k, S_k, Vt_k, shape_info = kpsvd(M, k, left_scale)
        print(f"KPSVD factors: U_k {U_k.shape}, S_k {S_k.shape}, Vt_k {Vt_k.shape}")

        print(f"Reconstructing k={k} approximation...")
        M_approx = reconstruct_from_kpsvd(U_k, S_k, Vt_k, shape_info)

        # Calculate compression ratio
        p, q, r, s = shape_info
        original_size = m * n
        compressed_size = p * q * k + k + r * s * k
        compression_ratio = original_size / compressed_size

        print(f"Compression ratio: {compression_ratio:.2f}×")

        print(f"Generating left factor noise series (levels: {noise_levels})...")
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

        print(f"Generating original downscale-upscale series (scales: {scale_factors})...")
        original_scale_images = []
        for scale in scale_factors:
            img = downscale_upscale_matrix(M, scale)
            original_scale_images.append(img)

        print(f"Generating right factor downscale-upscale series (scales: {scale_factors})...")
        right_factor_scale_images = []
        for scale in scale_factors:
            scaled_V = downscale_upscale_factor(Vt_k.T, scale, r, s)
            img = reconstruct_from_kpsvd(U_k, S_k, scaled_V.T, shape_info)
            right_factor_scale_images.append(img)

        all_k_results[k] = {
            'approximation': M_approx,
            'left_noise_images': left_noise_images,
            'right_noise_images': right_noise_images,
            'original_scale_images': original_scale_images,
            'right_factor_scale_images': right_factor_scale_images,
            'compression_ratio': compression_ratio
        }

    print(f"\nGenerating HTML visualization...")
    generate_html_visualization(M, all_k_results, noise_levels, scale_factors, shape_info, output_html)

    print("\nDone!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python kpsvd_demo.py <left_scale> <image_path> [k_values...] [--noise noise_levels...] [--scale scale_factors...]")
        print("Example: python kpsvd_demo.py 1 image.jpg 5 10 20 50 --noise 5 10 20 --scale 1.0 0.5 0.25 0.125")
        print("Default k values: [5, 10, 20, 50]")
        print("Default noise levels: [5, 10, 20]")
        print("Default scale factors: [1.0, 0.5, 0.25, 0.125]")
        sys.exit(1)

    left_scale = int(sys.argv[1])
    image_path = sys.argv[2]

    # Parse k values, noise levels, and scale factors
    k_values = []
    noise_levels = []
    scale_factors = []

    i = 3
    while i < len(sys.argv) and sys.argv[i] not in ['--noise', '--scale']:
        k_values.append(int(sys.argv[i]))
        i += 1

    if i < len(sys.argv) and sys.argv[i] == '--noise':
        i += 1
        while i < len(sys.argv) and sys.argv[i] != '--scale':
            noise_levels.append(float(sys.argv[i]))
            i += 1

    if i < len(sys.argv) and sys.argv[i] == '--scale':
        i += 1
        while i < len(sys.argv):
            scale_factors.append(float(sys.argv[i]))
            i += 1

    k_values = k_values if k_values else None
    noise_levels = noise_levels if noise_levels else None
    scale_factors = scale_factors if scale_factors else None

    main(image_path, left_scale, k_values, noise_levels, scale_factors)
