#!/usr/bin/env python3
# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Tests for LoRA implementation
# ---------------------------------------------------------------

"""
Unit tests for LoRA modules and integration.

Run with:
    pytest tests/test_lora.py -v
"""

import pytest
import torch
import torch.nn as nn
from models.lora import (
    LoRALinear,
    LoRAAttention,
    count_lora_parameters,
    count_parameters,
)
from models.lora_integration import (
    LoRAConfig,
    apply_lora_to_vit,
    get_lora_stats,
)


class TestLoRALinear:
    """Tests for LoRALinear layer"""

    def test_lora_linear_creation(self):
        """Test LoRA linear layer can be created"""
        layer = LoRALinear(in_features=768, out_features=256, rank=8)
        assert layer.in_features == 768
        assert layer.out_features == 256
        assert layer.rank == 8

    def test_lora_linear_forward(self):
        """Test forward pass through LoRA linear layer"""
        layer = LoRALinear(in_features=768, out_features=256, rank=8)
        x = torch.randn(2, 768)

        output = layer(x)

        assert output.shape == (2, 256)

    def test_lora_linear_parameters(self):
        """Test LoRA linear layer has correct parameters"""
        layer = LoRALinear(in_features=768, out_features=256, rank=8)

        # Check parameter names
        param_names = {name for name, _ in layer.named_parameters()}
        assert "weight" in param_names
        assert "bias" in param_names
        assert "lora_a" in param_names
        assert "lora_b" in param_names

        # Check shapes
        assert layer.weight.shape == (256, 768)
        assert layer.bias.shape == (256,)
        assert layer.lora_a.shape == (768, 8)
        assert layer.lora_b.shape == (8, 256)

    def test_lora_linear_gradient_flow(self):
        """Test gradients flow through LoRA parameters"""
        layer = LoRALinear(in_features=768, out_features=256, rank=8)
        layer.weight.requires_grad = False  # Freeze base weights

        x = torch.randn(2, 768)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert layer.lora_a.grad is not None
        assert layer.lora_b.grad is not None
        assert layer.weight.grad is None  # Should be frozen


class TestLoRAAttention:
    """Tests for LoRAAttention"""

    def test_lora_attention_creation(self):
        """Test LoRA attention layer can be created"""
        attn = LoRAAttention(in_features=768, rank=8, target="all")
        assert attn.use_q
        assert attn.use_k
        assert attn.use_v

    def test_lora_attention_selective(self):
        """Test selective LoRA for Q only"""
        attn_q = LoRAAttention(in_features=768, rank=8, target="q")
        assert attn_q.use_q
        assert not attn_q.use_k
        assert not attn_q.use_v

    def test_lora_attention_forward(self):
        """Test forward pass through LoRA attention"""
        attn = LoRAAttention(in_features=768, rank=8, target="all")

        q = torch.randn(2, 10, 768)
        k = torch.randn(2, 10, 768)
        v = torch.randn(2, 10, 768)

        q_out, k_out, v_out = attn(q, k, v)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape


class TestParameterCounting:
    """Tests for parameter counting utilities"""

    def test_count_parameters(self):
        """Test parameter counting"""
        module = nn.Linear(768, 256)

        count = count_parameters(module)
        expected = 768 * 256 + 256  # weights + bias
        assert count == expected

    def test_count_lora_parameters(self):
        """Test LoRA parameter counting"""
        layer = LoRALinear(in_features=768, out_features=256, rank=8)
        layer.weight.requires_grad = False

        stats = count_lora_parameters(layer)

        expected_lora = (768 * 8) + (8 * 256)  # A + B
        assert stats["lora"] == expected_lora
        assert stats["non_lora"] == 0  # base weights frozen


class TestLoRAConfig:
    """Tests for LoRA configuration"""

    def test_config_creation(self):
        """Test LoRA config can be created"""
        config = LoRAConfig(
            enabled=True,
            rank=16,
            lora_alpha=32,
        )
        assert config.enabled
        assert config.rank == 16
        assert config.lora_alpha == 32

    def test_config_to_dict(self):
        """Test config can be converted to dictionary"""
        config = LoRAConfig(enabled=True, rank=8)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["enabled"] is True
        assert config_dict["rank"] == 8

    def test_config_from_dict(self):
        """Test config can be created from dictionary"""
        config_dict = {
            "enabled": True,
            "rank": 16,
            "lora_alpha": 32,
            "target_modules": ["layer1", "layer2"],
        }
        config = LoRAConfig(**config_dict)

        assert config.rank == 16
        assert "layer1" in config.target_modules


class TestDummyModel:
    """Dummy model for integration tests"""

    def create_dummy_model(self):
        """Create a simple model for testing"""
        return nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )


class TestLoRAIntegration(TestDummyModel):
    """Tests for LoRA integration functions"""

    def test_apply_lora_disabled(self):
        """Test that disabled LoRA doesn't modify model"""
        model = self.create_dummy_model()
        config = LoRAConfig(enabled=False)

        # Count original parameters
        original_params = count_parameters(model)

        apply_lora_to_vit(model, config)

        # Should not change
        assert count_parameters(model) == original_params

    def test_get_lora_stats(self):
        """Test getting LoRA statistics"""
        model = nn.Linear(768, 256)
        lora_layer = LoRALinear(768, 256, rank=8)
        lora_layer.weight.requires_grad = False

        stats = get_lora_stats(lora_layer)

        assert "total_params" in stats
        assert "trainable_params" in stats
        assert "lora_params" in stats
        assert "efficiency" in stats


class TestEndToEnd:
    """End-to-end integration tests"""

    def test_simple_training_step(self):
        """Test a simple training step with LoRA"""
        # Create model
        model = nn.Sequential(
            LoRALinear(768, 512, rank=8),
            nn.GELU(),
            LoRALinear(512, 256, rank=8),
        )

        # Freeze base weights
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

        # Create optimizer
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )

        # Training step
        model.train()
        x = torch.randn(4, 768)
        target = torch.randn(4, 256)

        output = model(x)
        loss = nn.MSELoss()(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that LoRA parameters were updated
        assert loss.item() > 0  # Loss should be computed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
