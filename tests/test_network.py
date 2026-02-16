"""Tests for nn/network.py."""

import torch
from nn.config import Config
from nn.network import RPSNet


class TestRPSNet:
    def test_output_shapes(self):
        config = Config()
        model = RPSNet(config)
        x = torch.randn(1, 15, 6, 6)
        policy, value = model(x)

        assert policy.shape == (1, 144)
        assert value.shape == (1, 1)

    def test_batch_output_shapes(self):
        config = Config()
        model = RPSNet(config)
        batch_size = 8
        x = torch.randn(batch_size, 15, 6, 6)
        policy, value = model(x)

        assert policy.shape == (batch_size, 144)
        assert value.shape == (batch_size, 1)

    def test_value_range(self):
        """Value head should output in [-1, 1] due to tanh."""
        config = Config()
        model = RPSNet(config)
        x = torch.randn(32, 15, 6, 6)
        _, value = model(x)

        assert value.min().item() >= -1.0
        assert value.max().item() <= 1.0

    def test_no_nan(self):
        config = Config()
        model = RPSNet(config)
        x = torch.randn(4, 15, 6, 6)
        policy, value = model(x)

        assert not torch.isnan(policy).any()
        assert not torch.isnan(value).any()

    def test_gradient_flow(self):
        """Verify gradients flow through both heads."""
        config = Config()
        model = RPSNet(config)
        x = torch.randn(2, 15, 6, 6)
        policy, value = model(x)

        loss = policy.sum() + value.sum()
        loss.backward()

        # Check that input conv has gradients
        assert model.input_conv.weight.grad is not None
        assert model.input_conv.weight.grad.abs().sum() > 0
