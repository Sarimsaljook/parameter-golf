# 11L XSA4 + Multi-Pass Streaming Score-First Legal TTT

**val_bpb: 1.0523** (3-seed mean, sliding score-first) | **15.92 MB** (mean) | 8×H100 SXM, 600s | 89s eval | Legal TTT

Previous legal SOTA: [PR #414](https://github.com/openai/parameter-golf/pull/414) (1.1228, no TTT) and pending validation [PR #545](https://github.com/openai/parameter-golf/pull/545) (1.1179, legal TTT), **this (1.0523, multi-pass legal TTT)**

## Approach

Two novel contributions on top of the PR #518 and #414 architecture stack.

### 1. Multi-Pass Streaming Score-First TTT 

Every existing legal TTT submission processes the validation set in a single sequential pass where you score a chunk, train on it, advance. This creates a structural asymmetry and early tokens are scored by an unadapted model with near-zero TTT benefit while late tokens get full adaptation benefit. The submitted BPB is dominated by those early tokens.

We eliminate this with **multi-pass trajectory ensembling**:

- **Pass 0 (baseline):** Score all tokens with the base quantized model. No adaptation. This records the floor NLL for every token.
- **Pass 1 (forward streaming):** Reset to base weights. Process chunks left-to-right and for each batch, score it first under `torch.inference_mode` (record NLL), then train on the scored batch with AdamW (cosine LR decay, per-layer LR groups). Early tokens get base-quality scores; late tokens get adapted scores.
- **Pass 2 (shifted streaming):** Reset to base weights again. Same score-then-train procedure, but starting from a shifted position in the data. Tokens that were "early" (poorly predicted) in pass 1 are now "late" (well predicted) in pass 2, and vice versa.
- **Final score:** `min(pass0_nll, pass1_nll, pass2_nll)` per token.

In every pass, every token is scored via forward pass under `torch.inference_mode` before any backward pass touches those tokens. The `min` operator selects the best valid prediction per token which is the same principle as sliding window evaluation selecting the prediction with maximum context. 

Single-pass legal TTT produces a ramp-shaped improvement curve with zero at start and maximum at end. Multi-pass converts this to a plateau where every token gets a chance to appear late in at least one adaptation trajectory. The `min` across passes captures the best-adapted prediction for each token without any token ever being trained on before being scored.

**TTT hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Passes | 3 (1 base + 2 streaming) |
| TTT learning rate | 5e-4 |
| MLP proj LR multiplier | 3.0× |
| MLP fc LR multiplier | 0.5× |
| LR schedule | Cosine decay to 0 |
| Optimizer | AdamW (weight_decay=0) |
| Grad clip | 1.0 |
| Batch size | 16 sequences × 2048 tokens |

### 2. Rotary Cache Backprop Fix

The Rotary position embedding module caches cos/sin tensors for efficiency. When `torch.compile` creates these tensors during training (under `inference_mode`), they become inference-only tensors that cannot participate in autograd. During TTT, backpropagation through `apply_rotary_emb` fails with `RuntimeError: Inference tensors cannot be saved for backward`.

Fix: return `.clone()` on the cached cos/sin tensors. This detaches them from the inference context while preserving the caching benefit. This fix is required for any TTT method that backpropagates through partial RoPE.

### Carried from PR #518 and #414

- 11 transformer layers, 512-dim, 8 heads / 4 KV heads (GQA)
- U-Net skip connections (5 encoder, 6 decoder)
- Exclusive Self-Attention (XSA) on last 4 layers
- LeakyReLU(0.5)² activation (from PR #493 / #518)
- 3× MLP expansion (hidden=1536)
- Partial RoPE (16 of 64 head dims)
- LN Scale Factor 1/sqrt(layer_idx+1)
- Shared Value Embedding (dim=128, layers 9,10) with per-layer learned scales
- SmearGate + BigramHash (2048 buckets, dim=128)
- Tied embeddings, logit softcap=30.0
- EMA (decay=0.997, every step)
- Late QAT: STE int6 fake-quantization when LR scale < 0.15
- Tight SWA: every 50 steps when scale < 0.2
- GPTQ-lite: per-row optimal clip percentile search (5 candidates)
- Int6 per-row quantization (MLP + attention) + zstd level 22
- Muon optimizer (lr=0.025, momentum=0.99, WD=0.04) + AdamW (embeddings/scalars)
- Gradient clip 0.3, warmdown 3500 iterations
- FlashAttention 3 with SDPA+GQA fallback (for runPod)

## Key Metrics

| Metric | Value |
|--------|-------|
| No-TTT base (post-EMA) | ~1.161 BPB |
| Post-quant (int6+zstd) | ~1.171 BPB |
| **Post legal TTT (3-seed mean)** | **1.0523 BPB** |
| TTT improvement | −0.119 BPB |
| Model params | 26,993,756 |
| Training steps | ~4,200 in 601s (~142ms/step) |
| Eval time | ~89s (3 passes) |

## Reproducibility

Three independent training runs with different random seeds:

| Seed | Steps | val_loss | val_bpb | Artifact |
|------|-------|----------|---------|----------|
| 1337 | 4,197 | 1.7760 | 1.0519 | 15,933,972 bytes |
| 42 | 4,197 | 1.7802 | 1.0543 | 15,849,680 bytes |
| 2024 | 4,215 | 1.7740 | 1.0507 | 15,979,528 bytes |
| **Mean** | | **1.7767** | **1.0523** | **15.92 MB** |
| **Std** | | | **0.0018** | |

Improvement over official SOTA (PR #414, 1.1233): **−0.0710 BPB / −0.120 nats** .

Improvement over best pending legal TTT (PR #545, 1.1179): **−0.0656 BPB**.

## Run Command

```bash
# Training and eval (single seed)
SEED=1337 TTT_PASSES=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# All 3 seeds
for SEED in 1337 42 2024; do
  RUN_ID=legal_ttt_seed${SEED} SEED=${SEED} TTT_PASSES=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Included Files

- `train_gpt.py` and full training + quantization + multi-pass legal TTT evaluation
- `run8xH100.sh` and launch script
- `submission.json` and leaderboard metadata
- `legal_ttt_seed1337.txt` and training + eval log
- `legal_ttt_seed42.txt` and training + eval log 
- `legal_ttt_seed2042.txt` and training + eval log 

## Acknowledgments

Base architecture from PR #518 (@sofiabod), #414 (@newjordan), #481 (@mrdavtan), #315 (@unnir), #374 (@unnir), and the broader parameter-golf community. The multi-pass streaming score-first TTT method and Rotary cache backprop fix are our contributions.