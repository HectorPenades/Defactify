"""FFT computation utilities for frequency-domain analysis."""

import numpy as np
from PIL import Image
from typing import Union


def compute_fft_grayscale(image: Image.Image) -> np.ndarray:
    """
    Compute FFT magnitude spectrum of a grayscale image.

    Converts image to grayscale, computes 2D FFT, shifts zero-frequency
    to center, and returns log-scaled magnitude normalized to [0, 1].

    Args:
        image: PIL Image (any mode)

    Returns:
        Float32 array of shape (H, W), values in [0, 1]
    """
    gray = np.array(image.convert('L'), dtype=np.float32)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shifted))
    # Normalize to [0, 1]
    min_val, max_val = magnitude.min(), magnitude.max()
    if max_val > min_val:
        magnitude = (magnitude - min_val) / (max_val - min_val)
    return magnitude.astype(np.float32)


def compute_fft_perchannel(image: Image.Image) -> np.ndarray:
    """
    Compute FFT magnitude spectrum per RGB channel.

    Computes 2D FFT independently for R, G, B channels, shifts
    zero-frequency to center, and returns log-scaled magnitude
    normalized to [0, 1] per channel.

    Args:
        image: PIL Image (will be converted to RGB)

    Returns:
        Float32 array of shape (H, W, 3), values in [0, 1]
    """
    rgb = np.array(image.convert('RGB'), dtype=np.float32)
    h, w = rgb.shape[:2]
    result = np.zeros((h, w, 3), dtype=np.float32)

    for c in range(3):
        channel = rgb[:, :, c]
        fft = np.fft.fft2(channel)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shifted))
        min_val, max_val = magnitude.min(), magnitude.max()
        if max_val > min_val:
            magnitude = (magnitude - min_val) / (max_val - min_val)
        result[:, :, c] = magnitude.astype(np.float32)

    return result
