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
    """
    A Linear layer wrapped with Low-Rank Adaptation (LoRA).

    Instead of fine-tuning the massive pre-trained weight matrix W,
    we inject two small trainable matrices A and B such that:
    W_new = W_old + (B @ A) * scaling

    This drastically reduces the number of trainable parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        """
        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            rank: The inner dimension of matrices A and B. Lower rank = fewer params.
            lora_alpha: Scaling constant. The update is scaled by (alpha / rank).
                        This allows changing 'rank' without retuning learning rates.
            lora_dropout: Dropout probability applied to the input of the LoRA path.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        # Original weights (frozen backbone)
        # We store them as parameters but will typically set requires_grad=False externally
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if True else None

        # LoRA trainable parameters: The "update" matrices
        # Shape: A is (in, rank), B is (rank, out)
        self.lora_a = nn.Parameter(torch.empty(in_features, rank))
        self.lora_b = nn.Parameter(torch.empty(rank, out_features))

        # Dropout for regularization during training
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize weights.
        Crucial Detail:
          - lora_a is initialized with a random distribution.
          - lora_b is initialized to ZEROS.

        Why? This ensures that at step 0, (B @ A) is zero.
        The model starts exactly as the pre-trained model, preventing instability.
        """
        # Standard initialization for the base layer
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # LoRA specific initialization
        nn.init.normal_(self.lora_a, std=1 / self.rank)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass adding the LoRA adaptation term.

        Formula:
            Output = (W_base @ x) + (scaling * B @ A @ Dropout(x))
        """

        # 1. Compute output using the frozen pre-trained weights
        out = torch.nn.functional.linear(x, self.weight, self.bias)

        # 2. Compute the low-rank update (the "adapter" path)
        lora_out = self.lora_dropout(x) @ self.lora_a @ self.lora_b
        lora_out = lora_out * self.scaling

        # 3. Sum them up
        return out + lora_out


class LoRAAttention(nn.Module):
    """
    A helper module to apply LoRA specifically to Attention Q, K, V projections.
    Usually used if you are modifying an Attention block definition directly
    rather than replacing Linear layers genericallly.
    """

    def __init__(
        self,
        in_features: int,
        rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target: str = "all",  # Which projections to adapt: "q", "k", "v", or "all"
    ):
        super().__init__()

        self.in_features = in_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target = target

        # Determine which components get adapters
        self.use_q = target in ["q", "all"]
        self.use_k = target in ["k", "all"]
        self.use_v = target in ["v", "all"]

        # Initialize distinct LoRA matrices for each target (Q, K, V)
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
        """Helper to compute the (B @ A @ x) term."""
        return self.dropout(x) @ lora_a @ lora_b * self.scaling

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Receives original Q, K, V tensors and adds the LoRA update to them.
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
    """
    Recursively replaces `nn.Linear` layers in a model with `LoRALinear`.

    This allows injecting LoRA into an existing standard PyTorch model without
    rewriting the model class definition.

    Args:
        module: The root module (e.g., the whole ViT or a specific block).
        target_modules: List of specific names (e.g., 'qkv', 'fc1') to target.
                        If None, it blindly replaces ALL linear layers (use with caution).
    """

    if target_modules is None:
        # Strategy 1: Replace ALL Linear layers found recursively
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Instantiate replacement
                lora_linear = LoRALinear(
                    child.in_features,
                    child.out_features,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                # Transfer weights
                lora_linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    lora_linear.bias.data.copy_(child.bias.data)

                # Swap the layer
                setattr(module, name, lora_linear)
            else:
                # Recurse deeper
                replace_linear_with_lora(
                    child,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                )
    else:
        # Strategy 2: Replace only specific modules by path
        # Note: This implementation assumes simple attribute access.
        # For complex paths like "blocks[0].attn.qkv", a more robust traverser is usually needed.
        for target in target_modules:
            # Basic logic to traverse dot-separated paths
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
    """
    Freezes the 'weight' (base model) parameters in LoRA layers.
    Ensures only the adapter matrices (A and B) are trained.
    """
    for name, param in module.named_parameters():
        if "lora_" not in name and "weight" in name:
            param.requires_grad = False


def unfreeze_lora_params(module: nn.Module) -> None:
    """Unfreezes everything (useful for debugging or full fine-tuning)."""
    for param in module.parameters():
        param.requires_grad = True


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Utility to count total parameters."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


def count_lora_parameters(module: nn.Module) -> dict[str, int]:
    """
    Breaks down parameter counts into 'LoRA' vs 'Non-LoRA'.
    Useful to verify that LoRA adds very few parameters (~1-2% of total).
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
