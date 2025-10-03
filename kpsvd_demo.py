#!/usr/bin/env python3
"""
KPSVD Image Approximation with Noise Visualization
"""

import numpy as np
from PIL import Image
import argparse
from pathlib import Path


def load_and_convert_to_grayscale(image_path):
    """Load image and convert to grayscale"""
    img = Image.open(image_path)
    gray_img = img.convert('L')
    return np.array(gray_img, dtype=float)


def kpsvd(A, k1, k2):
    """
    Kronecker Product SVD (KPSVD) - Van Loan–Pitsianis method

    Approximates matrix A ≈ (U1 ⊗ U2) @ diag(s) @ (V1 ⊗ V2)^T
    using the rearrangement matrix R(A) approach.

    Args:
        A: Input matrix of shape (m, n)
        k1: Rank for first Kronecker dimension
        k2: Rank for second Kronecker dimension

    Returns:
        U1, U2: Left Kronecker factors
        S: Singular values
        V1, V2: Right Kronecker factors
    """
    m, n = A.shape

    # Determine factorization dimensions (try to find balanced factorization)
    m1 = int(np.sqrt(m))
    m2 = m // m1
    n1 = int(np.sqrt(n))
    n2 = n // n1

    # Adjust if not perfectly divisible
    if m1 * m2 != m:
        m1 = int(np.sqrt(m))
        m2 = m // m1
        while m1 * m2 != m and m1 > 1:
            m1 -= 1
            m2 = m // m1
        if m1 * m2 != m:
            m1, m2 = m, 1

    if n1 * n2 != n:
        n1 = int(np.sqrt(n))
        n2 = n // n1
        while n1 * n2 != n and n1 > 1:
            n1 -= 1
            n2 = n // n1
        if n1 * n2 != n:
            n1, n2 = n, 1

    # Van Loan–Pitsianis: Create rearrangement matrix R(A)
    # Reshape A into (m1, m2, n1, n2), then permute to (m1, n1, m2, n2)
    # and reshape to (m1*n1, m2*n2)
    A_reshaped = A.reshape(m1, m2, n1, n2)
    A_permuted = np.transpose(A_reshaped, (0, 2, 1, 3))
    R_A = A_permuted.reshape(m1 * n1, m2 * n2)

    # Perform truncated SVD on R(A)
    k = min(k1 * k2, min(R_A.shape))
    U_r, s_r, Vt_r = np.linalg.svd(R_A, full_matrices=False)
    U_r = U_r[:, :k]
    s_r = s_r[:k]
    Vt_r = Vt_r[:k, :]

    # Extract Kronecker factors from the rearranged SVD
    # Reshape U_r from (m1*n1, k) to (m1, n1, k) to extract U1 and V1
    U_r_reshaped = U_r.reshape(m1, n1, k)

    # Get U1 and V1 through averaging and SVD
    U1_init = np.mean(U_r_reshaped, axis=1)  # (m1, k)
    V1_init = np.mean(U_r_reshaped, axis=0)  # (n1, k)

    U1, _, _ = np.linalg.svd(U1_init, full_matrices=False)
    U1 = U1[:, :k1]
    V1, _, _ = np.linalg.svd(V1_init, full_matrices=False)
    V1 = V1[:, :k1]

    # Reshape Vt_r from (k, m2*n2) to (k, m2, n2) to extract U2 and V2
    Vt_r_reshaped = Vt_r.reshape(k, m2, n2)

    # Get U2 and V2 through averaging and SVD
    U2_init = np.mean(Vt_r_reshaped, axis=2).T  # (m2, k)
    V2_init = np.mean(Vt_r_reshaped, axis=1).T  # (n2, k)

    U2, _, _ = np.linalg.svd(U2_init, full_matrices=False)
    U2 = U2[:, :k2]
    V2, _, _ = np.linalg.svd(V2_init, full_matrices=False)
    V2 = V2[:, :k2]

    # Compute optimal singular values
    U_kron = np.kron(U1, U2)
    V_kron = np.kron(V1, V2)
    S_kron = U_kron.T @ A @ V_kron

    _, s_final, _ = np.linalg.svd(S_kron, full_matrices=False)

    return U1, U2, s_final, V1, V2


def kpsvd_approximation(U1, U2, S, V1, V2):
    """Reconstruct matrix from KPSVD factors"""
    U_kron = np.kron(U1, U2)
    V_kron = np.kron(V1, V2)
    k = min(len(S), U_kron.shape[1], V_kron.shape[1])
    return U_kron[:, :k] @ np.diag(S[:k]) @ V_kron[:, :k].T


def add_noise_to_factor(factor, noise_level):
    """Add Gaussian noise to a factor"""
    noise = np.random.randn(*factor.shape) * noise_level * np.std(factor)
    return factor + noise


def clip_to_image_range(img_array):
    """Clip array values to valid image range [0, 255]"""
    return np.clip(img_array, 0, 255).astype(np.uint8)


def generate_html(original, gray, approx, left_noise_series, right_noise_series,
                  k1, k2, noise_levels, output_path):
    """Generate HTML visualization"""

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>KPSVD Image Approximation with Noise</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        h2 {
            color: #666;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .image-item {
            text-align: center;
        }
        .image-item img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .image-item p {
            margin-top: 10px;
            color: #666;
            font-size: 14px;
        }
        .params {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        .params p {
            margin: 5px 0;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>KPSVD Image Approximation with Noise</h1>

        <div class="params">
            <p><strong>Kronecker Ranks:</strong> k1={k1}, k2={k2}</p>
            <p><strong>Noise levels:</strong> {noise_levels}</p>
        </div>

        <h2>Original and Approximation</h2>
        <div class="image-grid">
            <div class="image-item">
                <img src="{original}" alt="Original Image">
                <p>Original Image</p>
            </div>
            <div class="image-item">
                <img src="{gray}" alt="Grayscale Image">
                <p>Grayscale Image</p>
            </div>
            <div class="image-item">
                <img src="{approx}" alt="KPSVD Approximation">
                <p>KPSVD Approximation (k1={k1}, k2={k2})</p>
            </div>
        </div>

        <h2>Left Kronecker Factor (U1 ⊗ U2) Noise Series</h2>
        <div class="image-grid">
""".format(k1=k1, k2=k2, noise_levels=', '.join(map(str, noise_levels)),
           original=original, gray=gray, approx=approx)

    for i, (img_path, level) in enumerate(left_noise_series):
        html += f"""            <div class="image-item">
                <img src="{img_path}" alt="Left noise {level}">
                <p>Left Factor Noise: {level}</p>
            </div>
"""

    html += """        </div>

        <h2>Right Kronecker Factor (V1 ⊗ V2) Noise Series</h2>
        <div class="image-grid">
"""

    for i, (img_path, level) in enumerate(right_noise_series):
        html += f"""            <div class="image-item">
                <img src="{img_path}" alt="Right noise {level}">
                <p>Right Factor Noise: {level}</p>
            </div>
"""

    html += """        </div>
    </div>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description='KPSVD Image Approximation with Noise Visualization')
    parser.add_argument('image', type=str, help='Input image path')
    parser.add_argument('-k1', '--rank1', type=int, default=10, help='First Kronecker rank (default: 10)')
    parser.add_argument('-k2', '--rank2', type=int, default=10, help='Second Kronecker rank (default: 10)')
    parser.add_argument('-n', '--noise-levels', type=float, nargs='+',
                        default=[0.01, 0.05, 0.1, 0.2],
                        help='Noise levels to apply (default: 0.01 0.05 0.1 0.2)')
    parser.add_argument('-o', '--output', type=str, default='kpsvd_visualization.html',
                        help='Output HTML file (default: kpsvd_visualization.html)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    print(f"Loading image: {args.image}")
    original_img = Image.open(args.image)

    # Save original
    original_path = output_dir / 'original.png'
    original_img.save(original_path)

    # Convert to grayscale
    print("Converting to grayscale...")
    gray_array = load_and_convert_to_grayscale(args.image)
    gray_img = Image.fromarray(gray_array.astype(np.uint8))
    gray_path = output_dir / 'grayscale.png'
    gray_img.save(gray_path)

    # Apply KPSVD
    print(f"Applying KPSVD with ranks k1={args.rank1}, k2={args.rank2}...")
    U1, U2, S, V1, V2 = kpsvd(gray_array, args.rank1, args.rank2)

    # KPSVD approximation
    print("Creating KPSVD approximation...")
    approx_array = kpsvd_approximation(U1, U2, S, V1, V2)
    approx_img = Image.fromarray(clip_to_image_range(approx_array))
    approx_path = output_dir / f'approx_k1{args.rank1}_k2{args.rank2}.png'
    approx_img.save(approx_path)

    # Generate left factor noise series (noise on U1 and U2)
    print("Generating left Kronecker factor noise series...")
    left_noise_series = []
    for noise_level in args.noise_levels:
        U1_noisy = add_noise_to_factor(U1, noise_level)
        U2_noisy = add_noise_to_factor(U2, noise_level)
        noisy_array = kpsvd_approximation(U1_noisy, U2_noisy, S, V1, V2)
        noisy_img = Image.fromarray(clip_to_image_range(noisy_array))
        img_path = output_dir / f'left_noise_{noise_level}.png'
        noisy_img.save(img_path)
        left_noise_series.append((f'output/{img_path.name}', noise_level))

    # Generate right factor noise series (noise on V1 and V2)
    print("Generating right Kronecker factor noise series...")
    right_noise_series = []
    for noise_level in args.noise_levels:
        V1_noisy = add_noise_to_factor(V1, noise_level)
        V2_noisy = add_noise_to_factor(V2, noise_level)
        noisy_array = kpsvd_approximation(U1, U2, S, V1_noisy, V2_noisy)
        noisy_img = Image.fromarray(clip_to_image_range(noisy_array))
        img_path = output_dir / f'right_noise_{noise_level}.png'
        noisy_img.save(img_path)
        right_noise_series.append((f'output/{img_path.name}', noise_level))

    # Generate HTML
    print(f"Generating HTML visualization: {args.output}")
    generate_html(
        f'output/{original_path.name}',
        f'output/{gray_path.name}',
        f'output/{approx_path.name}',
        left_noise_series,
        right_noise_series,
        args.rank1,
        args.rank2,
        args.noise_levels,
        args.output
    )

    print(f"\nDone! Open {args.output} in a browser to view the results.")


if __name__ == '__main__':
    main()
