# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# LoRA (Low-Rank Adaptation) implementation based on:
# "LoRA: Low-Rank Adaptation of Large Language Models"
# https://arxiv.org/abs/2106.09685
# ---------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """Linear layer with LoRA (Low-Rank Adaptation) adaptation.

    Replaces a frozen linear layer with LoRA adaptation:
    output = W_0 @ x + (α/r) * (B @ A) @ x
    where A and B are trainable low-rank matrices.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        rank: Rank of the LoRA adaptation matrices
        lora_alpha: Scaling factor for LoRA output
        lora_dropout: Dropout probability for LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        # Original weights (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if True else None

        # LoRA parameters: output = W_0 @ x + (α/r) * B @ A @ x
        self.lora_a = nn.Parameter(torch.empty(in_features, rank))
        self.lora_b = nn.Parameter(torch.empty(rank, out_features))

        # Dropout for LoRA
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights following the LoRA paper."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize A with normal distribution, B with zeros
        nn.init.normal_(self.lora_a, std=1 / self.rank)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with original weights + LoRA adaptation.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Standard forward pass
        out = torch.nn.functional.linear(x, self.weight, self.bias)

        # LoRA adaptation
        lora_out = self.lora_dropout(x) @ self.lora_a @ self.lora_b
        lora_out = lora_out * self.scaling

        return out + lora_out


class LoRAAttention(nn.Module):
    """LoRA adaptation for attention layers.

    Applies LoRA to the Q, K, V projection layers in multi-head attention.
    """

    def __init__(
        self,
        in_features: int,
        rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target: str = "all",  # "q", "k", "v", or "all"
    ):
        super().__init__()

        self.in_features = in_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target = target

        # LoRA for Q, K, V projections
        self.use_q = target in ["q", "all"]
        self.use_k = target in ["k", "all"]
        self.use_v = target in ["v", "all"]

        if self.use_q:
            self.lora_q_a = nn.Parameter(torch.empty(in_features, rank))
            self.lora_q_b = nn.Parameter(torch.empty(rank, in_features))
            nn.init.normal_(self.lora_q_a, std=1 / rank)
            nn.init.zeros_(self.lora_q_b)

        if self.use_k:
            self.lora_k_a = nn.Parameter(torch.empty(in_features, rank))
            self.lora_k_b = nn.Parameter(torch.empty(rank, in_features))
            nn.init.normal_(self.lora_k_a, std=1 / rank)
            nn.init.zeros_(self.lora_k_b)

        if self.use_v:
            self.lora_v_a = nn.Parameter(torch.empty(in_features, rank))
            self.lora_v_b = nn.Parameter(torch.empty(rank, in_features))
            nn.init.normal_(self.lora_v_a, std=1 / rank)
            nn.init.zeros_(self.lora_v_b)

        self.scaling = lora_alpha / rank
        self.dropout = nn.Dropout(p=lora_dropout)

    def apply_lora(
        self, x: torch.Tensor, lora_a: nn.Parameter, lora_b: nn.Parameter
    ) -> torch.Tensor:
        """Apply LoRA adaptation to a tensor."""
        return self.dropout(x) @ lora_a @ lora_b * self.scaling

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply LoRA to Q, K, V tensors.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            Tuple of (q + lora_q, k + lora_k, v + lora_v)
        """
        if self.use_q:
            q = q + self.apply_lora(q, self.lora_q_a, self.lora_q_b)

        if self.use_k:
            k = k + self.apply_lora(k, self.lora_k_a, self.lora_k_b)

        if self.use_v:
            v = v + self.apply_lora(v, self.lora_v_a, self.lora_v_b)

        return q, k, v


def replace_linear_with_lora(
    module: nn.Module,
    rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[list[str]] = None,
) -> None:
    """Replace Linear layers in a module with LoRA layers.

    Args:
        module: The module to apply LoRA to
        rank: Rank of LoRA adaptation
        lora_alpha: Scaling factor
        lora_dropout: Dropout probability
        target_modules: List of module names to replace (e.g., ["mlp.fc1", "mlp.fc2"])
                       If None, replaces all Linear layers
    """

    if target_modules is None:
        # Replace all Linear layers
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                lora_linear = LoRALinear(
                    child.in_features,
                    child.out_features,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                # Copy original weights
                lora_linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    lora_linear.bias.data.copy_(child.bias.data)

                setattr(module, name, lora_linear)
            else:
                # Recursively apply to children
                replace_linear_with_lora(
                    child,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                )
    else:
        # Replace only specified modules
        for target in target_modules:
            parts = target.split(".")
            current = module
            for part in parts[:-1]:
                current = getattr(current, part)

            child = getattr(current, parts[-1])
            if isinstance(child, nn.Linear):
                lora_linear = LoRALinear(
                    child.in_features,
                    child.out_features,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                lora_linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    lora_linear.bias.data.copy_(child.bias.data)

                setattr(current, parts[-1], lora_linear)


def freeze_lora_params(module: nn.Module) -> None:
    """Freeze original weights in LoRA layers, keeping only adaptation parameters trainable."""
    for name, param in module.named_parameters():
        if "lora_" not in name and "weight" in name:
            param.requires_grad = False


def unfreeze_lora_params(module: nn.Module) -> None:
    """Unfreeze all parameters in LoRA layers."""
    for param in module.parameters():
        param.requires_grad = True


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a module.

    Args:
        module: The module to count parameters for
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


def count_lora_parameters(module: nn.Module) -> dict[str, int]:
    """Count LoRA adaptation parameters.

    Args:
        module: The module containing LoRA layers

    Returns:
        Dictionary with "lora" and "non_lora" counts
    """
    lora_params = 0
    non_lora_params = 0

    for name, param in module.named_parameters():
        if param.requires_grad:
            if "lora_" in name:
                lora_params += param.numel()
            else:
                non_lora_params += param.numel()

    return {
        "lora": lora_params,
        "non_lora": non_lora_params,
        "total_trainable": lora_params + non_lora_params,
    }
