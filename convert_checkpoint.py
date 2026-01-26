import torch
import torch.nn.functional as F

# --- CONFIGURA I PERCORSI ---
INPUT_CKPT = "/teamspace/studios/this_studio/epoch_106-step_19902_eomt.ckpt"
OUTPUT_CKPT = "/teamspace/studios/this_studio/epoch_106_patch16_resized_clean.ckpt" # Nota il nome _clean
KEY = "network.encoder.backbone.pos_embed"

print(f"Loading {INPUT_CKPT}...")
ckpt = torch.load(INPUT_CKPT, map_location="cpu")

# --- PASSO 1: Interpolazione Griglia (Patch 16: 4096 -> 3136) ---
pos_embed = ckpt["state_dict"][KEY]
n_tokens_old = pos_embed.shape[1]
size_old = int(n_tokens_old ** 0.5) # 64
n_tokens_new = 3136 # 896 / 16 = 56 -> 56^2
size_new = 56

print(f"Interpolazione: {size_old}x{size_old} -> {size_new}x{size_new}")
dim = pos_embed.shape[-1]
grid = pos_embed.reshape(1, size_old, size_old, dim).permute(0, 3, 1, 2)
new_grid = F.interpolate(grid, size=(size_new, size_new), mode='bicubic', align_corners=False)
new_pos_embed = new_grid.permute(0, 2, 3, 1).reshape(1, n_tokens_new, dim)
ckpt["state_dict"][KEY] = new_pos_embed

# --- PASSO 2: RIMOZIONE OTTIMIZZATORE (Il Fix per il tuo errore) ---
print("Rimozione stati ottimizzatore e scheduler vecchi...")
keys_to_remove = ["optimizer_states", "lr_schedulers", "loops", "callbacks"]
for k in keys_to_remove:
    if k in ckpt:
        del ckpt[k]
        print(f"  - Rimosso: {k}")

# Salvataggio
torch.save(ckpt, OUTPUT_CKPT)
print(f"✅ Fatto! Checkpoint pulito salvato in: {OUTPUT_CKPT}")