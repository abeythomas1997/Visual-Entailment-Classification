# Visual Entailment: From Classifier to Reasoning System

A two-part deep learning project building and improving a neural network that determines whether a natural language hypothesis is **consistent with** or **contradicted by** a given image.

---

## Part 1 — Building the Classifier

### Task & Dataset
Binary classification of image-text pairs: **entailment** (text matches image) or **contradiction** (text contradicts image). Dataset: 31,340 training / 3,922 validation / 3,867 test pairs, split by `image_id` to prevent leakage.

### Architecture
A dual-encoder model with cross-attention fusion:

- **Vision:** ViT-B/16 pretrained on ImageNet-21k → 196 × 768 patch tokens (fully frozen)
- **Text:** DistilBERT → 24 × 768 token embeddings (last 3 layers unfrozen)
- **Fusion:** Cross-attention block (text tokens attend to image patches) + LayerNorm + residual
- **Head:** Flatten → Dense(512, GELU) → Dropout(0.126) → Dense(1, logit)
- **Parameters:** 154.9M total — only 2.8M trainable

### Training Pipeline
| Phase | Description | Val F1 |
|-------|-------------|--------|
| 0 — SNLI-VE Warm-up | Pretrained fusion head on large external dataset; backbones frozen | — |
| 1 — Transfer to A2 | Trained head only on target dataset; 8 epochs, LR 1e-4 | ~75.8% |
| 2 — Augmentation | Added photometric jitter (brightness, contrast, saturation, hue) | ~76% |
| 3A — Text Fine-tuning | Unfroze 3 DistilBERT layers; ViT kept frozen; random HP search | **80.01%** |
| 4 — Vision + Text (ablation) | Unfroze 1 ViT block — degraded to 76–79% F1; discarded | — |

Best hyperparameters (random_05): LR 3.45e-5 · WD 1.33e-5 · Dropout 0.126 · Batch 32 · Epochs 2 · Threshold τ = 0.55

### Results
| Metric | Value |
|--------|-------|
| **Test Macro-F1** | **81.52%** ✓ (target ≥ 80%) |
| Test Accuracy | 81.59% |
| Generalisation gap | +1.51% F1 (val → test, positive generalisation) |
| Contradiction F1 | 82.60% (Recall 87.5%) |
| Entailment F1 | 80.45% (Recall 75.7%) |

Out-of-distribution robustness test (custom 50-pair dataset): **76% accuracy**, confirming solid transfer with known systematic gaps.

### Key Weaknesses Identified
Counting/quantifiers ("only one person") · Fine-grained actions (kneeling vs. standing) · Scene text/signage (no OCR) · Lexical surface bias · Subjective attributes ("smiling", "calm")

---

## Part 2 — Literature Survey & Proposed Improvements

Building on the Part 1 weaknesses, a literature survey identified five key papers and a concrete pipeline to push performance toward 84–86% F1.

### The Core Problem
The Part 1 model fails on **compositionality** — it cannot reason about counts, negations, or fine-grained spatial relations. This is a known gap in vision-language models documented by Thrush et al. (Winoground, CVPR 2022), who showed that models like CLIP and ALBEF perform near-random when two captions share identical words but different relations (e.g. *"mug in grass"* vs *"grass in mug"*).

### Proposed System: ALBEF + Three Targeted Enhancements

**Base model — ALBEF** (Li et al., NeurIPS 2021): Replaces the Part 1 ViT + DistilBERT stack with a model that *aligns before fusing* using joint ITC + ITM + MLM pretraining with momentum distillation. Establishes a stronger cross-modal baseline (83.14% on this task).

**Enhancement 1 — Diffusion-Based Semantic Augmentation** (Trabucco et al., ICML 2023): Text-guided diffusion edits generate hard negatives by adding, removing, or swapping objects in images — directly targeting the negation and spatial reasoning gaps. Expected gain: ~+10 pp on negation/spatial subsets.

**Enhancement 2 — Counting-Aware Contrastive Loss** (Paiss et al., ICCV 2023): An auxiliary counting head with counterfactual caption pairs (e.g. *"two dogs"* ↔ *"three dogs"*) forces the model to ground object quantities. Expected gain: ~+15% on counting F1.

**Enhancement 3 — LoRA Efficient Adaptation** (Hu et al., ICLR 2022): Low-rank adapters inserted into attention layers allow efficient reasoning fine-tuning with ~10,000× fewer trainable parameters than full fine-tuning — preventing overfitting while incorporating the new training signals.

### End-to-End Pipeline
```
ALBEF base → Diffusion Augmentation (hard negatives)
           → Counting-Aware Contrastive Loss (numeracy)
           → LoRA Fine-Tuning (efficient adaptation)
           → Self-Supervised Extension (unlabelled data scalability)
```

### Expected Performance

| System | Macro-F1 |
|--------|----------|
| Part 1 — Phase 3A (this repo) | 81.52% |
| ALBEF baseline | 83.14% |
| **Proposed system (all enhancements)** | **~84–86%** |
| Human performance | ~95% |

---

## Repository Structure

```
├── A2_data/                  # Images and JSONL annotation files
├── notebooks/
│   ├── phase0_snli_pretrain.ipynb
│   ├── phase1_transfer.ipynb
│   ├── phase3a_text_finetune.ipynb   ← best model
│   └── error_analysis.ipynb
├── models/                   # Saved checkpoints
├── A2_test_predictions.csv   # Final test set predictions
└── README.md
```

## Tech Stack
Python 3.10 · TensorFlow 2.x / Keras · Hugging Face Transformers (DistilBERT) · ViT-B/16 (ImageNet-21k) · AdamW · scikit-learn (t-SNE) · Diffusers (proposed)

## References
- Li et al. (2021). *Align before Fuse: ALBEF.* NeurIPS 2021.
- Trabucco et al. (2023). *Effective Data Augmentation with Diffusion Models.* ICML 2023.
- Thrush et al. (2022). *Winoground: Probing V-L Compositionality.* CVPR 2022.
- Paiss et al. (2023). *Teaching CLIP to Count to Ten.* ICCV 2023.
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words.* arXiv:2010.11929.
- Sanh et al. (2019). *DistilBERT.* arXiv:1910.01108.
