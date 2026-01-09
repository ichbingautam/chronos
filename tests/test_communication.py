"""Tests for communication layer."""

import pytest
import torch

from chronos.communication.sparse import (
    SignificanceFilter,
    SignificanceFilterConfig,
    compute_significance,
    compute_delta_norm,
)
from chronos.communication.compression import (
    topk_sparsify,
    topk_desparsify,
    quantize,
    dequantize,
    compress_gradients,
    decompress_gradients,
)


class TestSignificanceFilter:
    """Tests for SignificanceFilter."""

    def test_creation(self):
        """Test filter creation with default config."""
        filter = SignificanceFilter()

        assert filter.config.threshold == 0.01
        assert filter._accumulated_delta is None

    def test_should_communicate_significant(self):
        """Test that significant updates trigger communication."""
        config = SignificanceFilterConfig(threshold=0.01)
        filter = SignificanceFilter(config)

        # Large delta relative to params
        delta = {"w": torch.ones(10) * 0.5}
        params = {"w": torch.ones(10)}

        should_comm = filter.should_communicate(delta, params)

        # Delta norm / param norm = 0.5 > 0.01
        assert should_comm is True

    def test_should_communicate_insignificant(self):
        """Test that small updates don't trigger communication."""
        config = SignificanceFilterConfig(threshold=0.1, max_accumulation_steps=1000)
        filter = SignificanceFilter(config)

        # Small delta relative to params
        delta = {"w": torch.ones(10) * 0.001}
        params = {"w": torch.ones(10) * 10}

        should_comm = filter.should_communicate(delta, params)

        # Delta norm / param norm is small
        assert should_comm is False

    def test_forced_sync(self):
        """Test forced sync after max accumulation steps."""
        config = SignificanceFilterConfig(
            threshold=1.0,  # Very high - won't trigger naturally
            max_accumulation_steps=3
        )
        filter = SignificanceFilter(config)

        delta = {"w": torch.ones(5) * 0.001}
        params = {"w": torch.ones(5) * 100}

        # First two - should not trigger
        assert filter.should_communicate(delta, params) is False
        assert filter.should_communicate(delta, params) is False

        # Third - forced sync
        assert filter.should_communicate(delta, params) is True

    def test_accumulation(self):
        """Test gradient accumulation."""
        filter = SignificanceFilter()

        delta1 = {"w": torch.ones(5)}
        delta2 = {"w": torch.ones(5) * 2}
        params = {"w": torch.ones(5) * 100}

        filter._accumulate(delta1)
        filter._accumulate(delta2)

        accumulated = filter.get_accumulated_delta()

        assert torch.allclose(accumulated["w"], torch.ones(5) * 3)

    def test_reset(self):
        """Test reset after communication."""
        filter = SignificanceFilter()

        delta = {"w": torch.ones(5)}
        params = {"w": torch.ones(5)}

        filter.should_communicate(delta, params)
        filter.reset()

        assert filter._accumulated_delta is None
        assert filter._steps_since_sync == 0


class TestComputeSignificance:
    """Tests for significance computation."""

    def test_basic_significance(self):
        """Test basic significance calculation."""
        delta = {"w": torch.ones(10)}
        params = {"w": torch.ones(10) * 10}

        sig = compute_significance(delta, params)

        # ||delta|| / ||params|| = sqrt(10) / sqrt(1000) ≈ 0.1
        assert 0.09 < sig < 0.11

    def test_zero_params(self):
        """Test with near-zero params."""
        delta = {"w": torch.ones(5)}
        params = {"w": torch.zeros(5)}

        sig = compute_significance(delta, params)

        assert sig == float("inf")


class TestTopKSparsification:
    """Tests for top-k sparsification."""

    def test_basic_sparsify(self):
        """Test basic sparsification."""
        tensor = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0])

        values, indices, shape = topk_sparsify(tensor, k=2)

        assert len(values) == 2
        # Should have the two largest: 5.0 and 8.0
        assert 5.0 in values.tolist()
        assert 8.0 in values.tolist()

    def test_ratio_sparsify(self):
        """Test sparsification with ratio."""
        tensor = torch.randn(100)

        values, indices, shape = topk_sparsify(tensor, ratio=0.1)

        assert len(values) == 10  # 10% of 100

    def test_roundtrip(self):
        """Test sparsify -> desparsify roundtrip."""
        tensor = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0])

        values, indices, shape = topk_sparsify(tensor, k=2)
        reconstructed = topk_desparsify(values, indices, shape)

        # Should have only the top-k values
        assert reconstructed.shape == tensor.shape
        # Non-top-k should be zero
        assert (reconstructed == 0).sum() == 3


class TestQuantization:
    """Tests for quantization."""

    def test_fp32_passthrough(self):
        """Test FP32 quantization (no-op)."""
        tensor = torch.randn(10)

        quantized, scale, zp = quantize(tensor, bits=32)

        assert torch.allclose(quantized, tensor)

    def test_fp16(self):
        """Test FP16 quantization."""
        tensor = torch.randn(10)

        quantized, scale, zp = quantize(tensor, bits=16)

        assert quantized.dtype == torch.float16

        dequantized = dequantize(quantized, scale, zp)
        assert torch.allclose(dequantized, tensor, atol=1e-3)

    def test_int8(self):
        """Test INT8 quantization."""
        tensor = torch.tensor([0.0, 0.5, 1.0])

        quantized, scale, zp = quantize(tensor, bits=8)

        assert quantized.dtype == torch.uint8

        dequantized = dequantize(quantized, scale, zp)
        # INT8 has some error
        assert torch.allclose(dequantized, tensor, atol=0.01)


class TestGradientCompression:
    """Tests for full gradient compression."""

    def test_compress_decompress(self):
        """Test full compression -> decompression."""
        grads = {
            "layer1": torch.randn(100),
            "layer2": torch.randn(50, 50),
        }

        compressed, error, stats = compress_gradients(
            grads, topk_ratio=0.1, quantize_bits=32
        )

        assert "layer1" in compressed
        assert "layer2" in compressed
        assert stats.sparsity > 0.8  # Should be ~90% sparse

        decompressed = decompress_gradients(compressed)

        assert "layer1" in decompressed
        assert decompressed["layer1"].shape == grads["layer1"].shape
