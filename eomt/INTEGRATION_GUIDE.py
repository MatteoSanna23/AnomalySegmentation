#!/usr/bin/env python3
# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# LoRA Integration Checklist & Next Steps
# ---------------------------------------------------------------

"""
Integration Checklist for LoRA in your training pipeline.

Follow these steps to integrate LoRA into your existing training code.
"""

# ============================================================================
# STEP 1: VERIFY INSTALLATION
# ============================================================================


def step_1_verify_installation():
    """Verify that LoRA modules can be imported."""
    print("Step 1: Verifying installation...")

    try:
        from models.lora import LoRALinear, LoRAAttention
        from models.lora_integration import LoRAConfig, apply_lora_to_vit
        from models.eomt import EoMT

        print("✅ All LoRA modules imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


# ============================================================================
# STEP 2: CREATE LORA CONFIGURATION
# ============================================================================


def step_2_create_lora_config():
    """Create a LoRA configuration."""
    print("\nStep 2: Creating LoRA configuration...")

    from models.lora_integration import LoRAConfig

    # Option 1: Minimal configuration
    config_minimal = LoRAConfig(enabled=True)
    print("✅ Minimal config created")

    # Option 2: Recommended configuration
    config_recommended = LoRAConfig(
        enabled=True,
        rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["class_head", "mask_head"],
        freeze_base_model=True,
    )
    print("✅ Recommended config created")

    # Option 3: Aggressive fine-tuning
    config_aggressive = LoRAConfig(
        enabled=True,
        rank=16,
        lora_alpha=32,
        target_modules=["class_head", "mask_head", "upscale"],
        freeze_base_model=True,
    )
    print("✅ Aggressive config created")

    return config_recommended


# ============================================================================
# STEP 3: CREATE MODEL WITH LORA
# ============================================================================


def step_3_create_model_with_lora(lora_config):
    """Create an EoMT model with LoRA."""
    print("\nStep 3: Creating model with LoRA...")

    from models.vit import ViT
    from models.eomt import EoMT
    from models.lora_integration import print_lora_summary

    # Create encoder
    vit_encoder = ViT(
        img_size=(640, 640),
        backbone_name="vit_large_patch14_reg4_dinov2",
    )
    print("✅ ViT encoder created")

    # Create model with LoRA
    model = EoMT(
        encoder=vit_encoder,
        num_classes=19,
        num_q=100,
        lora_config=lora_config,
    )
    print("✅ EoMT model with LoRA created")

    # Print summary
    print_lora_summary(model, lora_config)

    return model


# ============================================================================
# STEP 4: INTEGRATE WITH LIGHTNING
# ============================================================================


def step_4_lightning_integration():
    """Show how to integrate with PyTorch Lightning."""
    print("\nStep 4: Lightning Integration")

    code = """
    # In your LightningModule.__init__():
    
    from models.lora_integration import LoRAConfig
    from models.eomt import EoMT
    from models.vit import ViT
    
    # Create LoRA config
    lora_config = LoRAConfig(
        enabled=True,
        rank=8,
        target_modules=["class_head", "mask_head"],
        freeze_base_model=True,
    )
    
    # Create model with LoRA
    vit = ViT(img_size=(640, 640))
    self.network = EoMT(
        encoder=vit,
        num_classes=19,
        num_q=100,
        lora_config=lora_config,
    )
    
    # In configure_optimizers():
    optimizer = AdamW(
        [p for p in self.network.parameters() if p.requires_grad],
        lr=self.hparams.lr
    )
    return optimizer
    """

    print(code)
    return code


# ============================================================================
# STEP 5: MONITOR TRAINING
# ============================================================================


def step_5_monitoring():
    """Show how to monitor LoRA during training."""
    print("\nStep 5: Monitoring LoRA Parameters")

    code = """
    # Log LoRA statistics to W&B
    from models.lora_integration import get_lora_stats
    
    def on_train_start(self):
        stats = get_lora_stats(self.network)
        self.logger.log_hyperparams(stats)
    
    # In training_step:
    def training_step(self, batch, batch_idx):
        ...
        self.log("train_loss", loss)
        
        # Log learning rate
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"])
        
        return loss
    """

    print(code)
    return code


# ============================================================================
# STEP 6: SAVE AND LOAD
# ============================================================================


def step_6_save_load():
    """Show how to save and load LoRA weights."""
    print("\nStep 6: Saving and Loading LoRA Weights")

    code = """
    import torch
    from models.lora_integration import get_lora_stats
    
    # Save only LoRA weights
    def save_lora_weights(model, path):
        lora_state = {
            name: param for name, param in model.named_parameters() 
            if "lora_" in name and param.requires_grad
        }
        torch.save(lora_state, path)
    
    # Load LoRA weights
    def load_lora_weights(model, path):
        lora_state = torch.load(path)
        model.load_state_dict(lora_state, strict=False)
    
    # Usage
    save_lora_weights(model, "checkpoint/lora_weights.pt")
    load_lora_weights(model, "checkpoint/lora_weights.pt")
    """

    print(code)
    return code


# ============================================================================
# INTEGRATION CHECKLIST
# ============================================================================

CHECKLIST = """
╔════════════════════════════════════════════════════════════════════════════╗
║                      LoRA INTEGRATION CHECKLIST                            ║
╚════════════════════════════════════════════════════════════════════════════╝

IMPLEMENTATION:
  ☐ Read LORA_QUICKSTART.md for quick overview
  ☐ Review LORA_README.md for detailed documentation
  ☐ Check lora_examples.py for usage examples
  ☐ Examine example_lightning_integration.py for complete example

INTEGRATION:
  ☐ Import LoRAConfig in your training code
  ☐ Create LoRA configuration
  ☐ Pass lora_config to EoMT model initialization
  ☐ Update configure_optimizers() to use trainable parameters
  ☐ Test model forward pass
  ☐ Verify parameter counts with print_lora_summary()

TRAINING:
  ☐ Start training with LoRA enabled
  ☐ Monitor training loss and metrics
  ☐ Verify only LoRA parameters are being updated
  ☐ Compare results with baseline (LoRA disabled)
  ☐ Adjust rank if needed based on results

EVALUATION:
  ☐ Save checkpoint with LoRA weights
  ☐ Evaluate on validation set
  ☐ Compare performance vs baseline
  ☐ Document results

DEPLOYMENT:
  ☐ Save LoRA weights as separate checkpoint
  ☐ Load LoRA weights for inference
  ☐ Optionally merge LoRA weights into base model
  ☐ Document LoRA configuration used

OPTIONAL:
  ☐ Run test suite: pytest tests/test_lora.py
  ☐ Experiment with different rank values
  ☐ Try different target_modules
  ☐ Profile memory usage
  ☐ Benchmark training speed

╔════════════════════════════════════════════════════════════════════════════╗
║                       KEY FILES TO REFERENCE                               ║
╚════════════════════════════════════════════════════════════════════════════╝

📖 Documentation:
   - LORA_README.md                    - Complete documentation
   - LORA_QUICKSTART.md                - 60-second quick start
   - IMPLEMENTATION_SUMMARY.md         - This implementation summary

💡 Examples:
   - lora_examples.py                  - 5 practical examples
   - example_lightning_integration.py  - Full Lightning module

🔧 Implementation:
   - models/lora.py                    - Core LoRA classes
   - models/lora_integration.py        - Integration utilities
   - models/eomt.py                    - Modified EoMT model

✅ Tests:
   - tests/test_lora.py                - Test suite

╔════════════════════════════════════════════════════════════════════════════╗
║                         MINIMAL WORKING EXAMPLE                            ║
╚════════════════════════════════════════════════════════════════════════════╝

from models.lora_integration import LoRAConfig
from models.eomt import EoMT
from models.vit import ViT

# Step 1: Create LoRA config
lora_config = LoRAConfig(
    enabled=True,
    rank=8,
    lora_alpha=16,
)

# Step 2: Create model
vit = ViT(img_size=(640, 640))
model = EoMT(
    encoder=vit,
    num_classes=19,
    num_q=100,
    lora_config=lora_config,
)

# Step 3: Use in training loop
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)

# That's it! LoRA is now active.
# Only LoRA parameters (~1M) will be trained, not the base model (270M).

╔════════════════════════════════════════════════════════════════════════════╗
║                           COMMON CONFIGURATIONS                            ║
╚════════════════════════════════════════════════════════════════════════════╝

1. CONSERVATIVE (Lower rank, fewer parameters):
   rank=4, lora_alpha=8, target_modules=["class_head"]
   → 400K trainable parameters

2. RECOMMENDED (Balanced):
   rank=8, lora_alpha=16, target_modules=["class_head", "mask_head"]
   → 1.1M trainable parameters

3. AGGRESSIVE (Higher rank, more capacity):
   rank=16, lora_alpha=32, target_modules=["class_head", "mask_head", "upscale"]
   → 2.5M trainable parameters

4. FULL MODEL (Train everything):
   LoRAConfig(enabled=True, freeze_base_model=False)
   → ~270M trainable parameters (but still memory efficient)

╔════════════════════════════════════════════════════════════════════════════╗
║                          PARAMETER RECOMMENDATIONS                         ║
╚════════════════════════════════════════════════════════════════════════════╝

rank:
  - Start with 8 (good balance)
  - Use 4-6 for constrained resources
  - Use 16-32 if results are suboptimal

lora_alpha:
  - General rule: lora_alpha = 2 * rank
  - Range: rank / 2 to 4 * rank

lora_dropout:
  - Default: 0.1 (10%)
  - Increase to 0.2 if overfitting
  - Decrease to 0.05 if underfitting

freeze_base_model:
  - True: Fine-tune only LoRA (recommended for resource-constrained)
  - False: Fine-tune everything (requires more memory)

╔════════════════════════════════════════════════════════════════════════════╗
║                            NEXT STEPS                                      ║
╚════════════════════════════════════════════════════════════════════════════╝

1. Read LORA_QUICKSTART.md (2 min)
2. Run lora_examples.py to see it in action (5 min)
3. Integrate into your training code (15 min)
4. Run your first training with LoRA (varies)
5. Compare results with baseline

For questions, see LORA_README.md or consult the examples.

Good luck with your LoRA fine-tuning! 🚀
"""


if __name__ == "__main__":
    print(CHECKLIST)

    # Uncomment to run step-by-step:
    # step_1_verify_installation()
    # config = step_2_create_lora_config()
    # model = step_3_create_model_with_lora(config)
    # step_4_lightning_integration()
    # step_5_monitoring()
    # step_6_save_load()
