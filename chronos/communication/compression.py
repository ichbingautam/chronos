"""
Gradient Compression Utilities.

Provides top-k sparsification and quantization to reduce
communication bandwidth in distributed training.

Key methods:
- Top-k sparsification: Only send k largest gradients
- Quantization: Reduce precision (FP32 -> FP16/INT8)
- Error feedback: Accumulate compression error for next round
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class CompressionStats:
    """Statistics from compression operation."""

    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    num_elements: int
    num_nonzero: int
    sparsity: float


def topk_sparsify(
    tensor: Tensor,
    k: Optional[int] = None,
    ratio: float = 0.01
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Keep only top-k elements by magnitude.

    Args:
        tensor: Input tensor to sparsify
        k: Number of elements to keep (overrides ratio)
        ratio: Fraction of elements to keep if k not specified

    Returns:
        Tuple of (values, indices, original_shape)
    """
    original_shape = tensor.shape
    flat = tensor.flatten()

    if k is None:
        k = max(1, int(len(flat) * ratio))
    k = min(k, len(flat))

    # Get top-k by absolute value
    _, indices = torch.topk(flat.abs(), k, sorted=False)
    values = flat[indices]

    return values, indices, torch.tensor(original_shape)


def topk_desparsify(
    values: Tensor,
    indices: Tensor,
    shape: Tensor
) -> Tensor:
    """
    Reconstruct dense tensor from sparse representation.

    Args:
        values: Non-zero values
        indices: Indices of non-zero values
        shape: Original tensor shape

    Returns:
        Reconstructed dense tensor
    """
    shape_tuple = tuple(shape.tolist())
    flat = torch.zeros(
        int(torch.prod(shape).item()),
        dtype=values.dtype,
        device=values.device
    )
    flat[indices] = values
    return flat.view(shape_tuple)


def quantize(
    tensor: Tensor,
    bits: int = 8,
    stochastic: bool = False
) -> Tuple[Tensor, float, float]:
    """
    Quantize tensor to reduced precision.

    Args:
        tensor: Input tensor
        bits: Number of bits for quantization (8, 16, or 32)
        stochastic: Use stochastic rounding

    Returns:
        Tuple of (quantized_tensor, scale, zero_point)
    """
    if bits == 32:
        return tensor.clone(), 1.0, 0.0

    if bits == 16:
        return tensor.half(), 1.0, 0.0

    # INT8 quantization
    min_val = tensor.min().item()
    max_val = tensor.max().item()

    # Compute scale and zero point
    qmin, qmax = 0, 255  # For uint8
    scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
    zero_point = qmin - min_val / scale if scale != 0 else 0

    # Quantize
    if stochastic:
        # Add random noise before rounding
        noise = torch.rand_like(tensor)
        quantized = torch.clamp(
            torch.floor((tensor / scale + zero_point) + noise),
            qmin, qmax
        ).to(torch.uint8)
    else:
        quantized = torch.clamp(
            torch.round(tensor / scale + zero_point),
            qmin, qmax
        ).to(torch.uint8)

    return quantized, scale, zero_point


def dequantize(
    quantized: Tensor,
    scale: float,
    zero_point: float
) -> Tensor:
    """
    Dequantize tensor back to float.

    Args:
        quantized: Quantized tensor
        scale: Quantization scale
        zero_point: Quantization zero point

    Returns:
        Dequantized float tensor
    """
    if quantized.dtype == torch.float32:
        return quantized

    if quantized.dtype == torch.float16:
        return quantized.float()

    # INT8 dequantization
    return (quantized.float() - zero_point) * scale


def compress_gradients(
    grads: Dict[str, Tensor],
    topk_ratio: float = 0.01,
    quantize_bits: int = 32,
    error_feedback: Optional[Dict[str, Tensor]] = None
) -> Tuple[Dict[str, any], Dict[str, Tensor], CompressionStats]:
    """
    Compress gradient dictionary for communication.

    Args:
        grads: Dictionary of gradients to compress
        topk_ratio: Fraction of elements to keep
        quantize_bits: Quantization bits (8, 16, or 32)
        error_feedback: Previous compression errors to add

    Returns:
        Tuple of (compressed_data, new_error_feedback, stats)
    """
    compressed = {}
    new_error = {}
    total_original = 0
    total_compressed = 0
    total_elements = 0
    total_nonzero = 0

    for name, grad in grads.items():
        # Add error feedback
        if error_feedback and name in error_feedback:
            grad = grad + error_feedback[name]

        original_bytes = grad.numel() * grad.element_size()
        total_original += original_bytes
        total_elements += grad.numel()

        # Top-k sparsification
        values, indices, shape = topk_sparsify(grad, ratio=topk_ratio)
        total_nonzero += len(values)

        # Quantization
        q_values, scale, zp = quantize(values, bits=quantize_bits)

        # Compute error feedback
        reconstructed = topk_desparsify(values, indices, shape)
        new_error[name] = grad - reconstructed

        # Store compressed representation
        compressed[name] = {
            "values": q_values,
            "indices": indices,
            "shape": shape,
            "scale": scale,
            "zero_point": zp,
            "dtype": str(grad.dtype),
        }

        # Estimate compressed size
        compressed_bytes = (
            q_values.numel() * q_values.element_size() +
            indices.numel() * indices.element_size() +
            shape.numel() * shape.element_size() +
            16  # metadata
        )
        total_compressed += compressed_bytes

    stats = CompressionStats(
        original_bytes=total_original,
        compressed_bytes=total_compressed,
        compression_ratio=total_original / max(total_compressed, 1),
        num_elements=total_elements,
        num_nonzero=total_nonzero,
        sparsity=1 - total_nonzero / max(total_elements, 1)
    )

    return compressed, new_error, stats


def decompress_gradients(
    compressed: Dict[str, any],
    device: Optional[torch.device] = None
) -> Dict[str, Tensor]:
    """
    Decompress gradient dictionary.

    Args:
        compressed: Compressed gradient data
        device: Device to place tensors on

    Returns:
        Dictionary of decompressed gradients
    """
    decompressed = {}

    for name, data in compressed.items():
        # Dequantize
        values = dequantize(data["values"], data["scale"], data["zero_point"])

        # Desparsify
        tensor = topk_desparsify(values, data["indices"], data["shape"])

        if device:
            tensor = tensor.to(device)

        decompressed[name] = tensor

    return decompressed
