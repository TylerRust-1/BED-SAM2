import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch

def get_adaptive_kernel_sizes(image_shape, min_k=5, max_k=151, num_scales=6, scale_factor=0.08):
    """
    Generate a list of Gaussian kernel sizes & sigmas for SOD.
    
    - Kernels scale with resolution (up to ~8% of longest side).
    - Uses log-spaced sizes for better multi-scale coverage.
    - Sigma set so kernel spans ~3σ.
    """
    H, W = image_shape[:2]
    base_dim = max(H, W)

    # Larger maximum kernel to capture global context
    max_kernel = int(scale_factor * base_dim)
    max_kernel = min(max_k, max(min_k, max_kernel | 1))  # force odd

    # Log-spaced kernel sizes (better multi-scale coverage than linear)
    kernel_sizes = np.geomspace(min_k, max_kernel, num_scales).astype(int)
    kernel_sizes = sorted({k | 1 for k in kernel_sizes})  # force odd + unique

    # Sigma rule: kernel covers ~3σ
    sigmas = [max(0.3, k / 6.0) for k in kernel_sizes]

    return kernel_sizes, sigmas


def multiscale_soft_edges(image, kernel_sizes, sigmas):
    """
    Compute soft multi-scale edges using Sobel gradient magnitude 
    after Gaussian blur. Returns a float32 edge map.
    """
    # Convert to numpy grayscale float32
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif torch.is_tensor(image):
        image = image.squeeze().cpu().numpy()

    image = image.astype(np.float32)
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image /= image.max() + 1e-8

    combined_edges = np.zeros_like(image, dtype=np.float32)

    for k, sigma in zip(kernel_sizes, sigmas):
        sobel_ksize = 3 

        blurred = cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=sobel_ksize)

        combined_edges += cv2.magnitude(grad_x, grad_y)

    combined_edges /= combined_edges.max() +1e-8
    return combined_edges

def shift_depth(depth_img):
    shifted = abs(depth_img.astype(float) - 127) * (255.0 / 128.0)
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)
    return shifted

def saliency_depth_calc(depth: Image.Image, rgb_image: Image.Image, gt: Image.Image):
    depth_np = np.array(depth, dtype=np.uint8)
    inverted_depth = 255 - depth_np
    depth_diff = shift_depth(depth_np)

    kernel_sizes, sigmas = get_adaptive_kernel_sizes(depth_np.shape)

    # Multiscale Sobel edges
    depth_edges = multiscale_soft_edges(depth_np, kernel_sizes, sigmas)
    depth_inv_edges = multiscale_soft_edges(inverted_depth, kernel_sizes, sigmas)
    depth_diff_edges = multiscale_soft_edges(depth_diff, kernel_sizes, sigmas)
   
    comb_depth_edges = np.maximum.reduce([depth_edges, depth_inv_edges, depth_diff_edges])
    combined_edges = comb_depth_edges / (comb_depth_edges.max() + 1e-8)
    
    # plt.figure(figsize=(12, 6))
    # plt.subplot(3, 3, 1)
    # plt.imshow(rgb_image, cmap='gray')
    # plt.title("RGB Image")

    # plt.subplot(3, 3, 2)
    # plt.imshow(gt, cmap='gray')
    # plt.title("Ground Truth")

    # plt.subplot(3, 3, 3)
    # plt.imshow(combined_edges, cmap='gray')
    # plt.title("Combined Edges")

    # plt.subplot(3, 3, 4)
    # plt.imshow(depth, cmap='gray')
    # plt.title("Depth Image")

    # plt.subplot(3, 3, 5)
    # plt.imshow(inverted_depth, cmap='gray')
    # plt.title("Inverse Depth")

    # plt.subplot(3, 3, 6)
    # plt.imshow(depth_diff, cmap='gray')
    # plt.title("Centered Depth")

    # plt.subplot(3, 3, 7)
    # plt.imshow(depth_edges, cmap='gray')
    # plt.title("Depth Edges")

    # plt.subplot(3, 3, 8)
    # plt.imshow(depth_inv_edges, cmap='gray')
    # plt.title("Inverse Depth Edges")

    # plt.subplot(3, 3, 9)
    # plt.imshow(depth_diff_edges, cmap='gray')
    # plt.title("Centered Depth Edges")

    # plt.tight_layout()
    # plt.show()

    combined_edges = (combined_edges * 255).astype(np.uint8)  # Scale edges to 0-255 for saving

    return Image.fromarray(combined_edges)
