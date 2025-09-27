# BERT vs T5 for NER: How Evaluation Choices Flip Rankings

**Goal.** This repo reproduces a small but robust NER evaluation comparing BERT (token classification) vs. T5 (seq2seq) across:
- **Evaluation lenses:** token macro-F1 (excluding `O`), labelled vs. unlabelled span F1.
- **Tagging variants:** full BIO & (optionally) simplified BIO.
- **Domains:** in-domain (EWT) vs. OOD (PUD).
- **Diagnostics:** boundary vs. type errors, entity length buckets, and rarity buckets.
- **Reliability:** paired bootstrap for span-F1; (optionally) 3-seed variance.

**Paper fit.** This is aimed at **Eval4NLP** (evaluation-focused) short paper.

---

## Data
We use **UniversalNER** English EWT and PUD IOB2 splits. The notebook downloads these automatically from the public GitHub mirrors.

> **Note on private data:** Our separate Amazon review dataset used in coursework is **not** included and is not required for this paper.

---

## Environment
- Python ≥ 3.10
- (Recommended) NVIDIA GPU with CUDA; CPU works but is slow for T5.
- See `requirements.txt` for suggested versions.

Add this cell at the **top** of the notebook to snapshot versions:
```python
import sys, torch, transformers, pandas as pd
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda, "| is_cuda:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("Pandas:", pd.__version__)

