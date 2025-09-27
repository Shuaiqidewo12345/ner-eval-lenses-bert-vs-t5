# BERT vs T5 for NER: How Evaluation Choices Flip Rankings

**Goal.** I reproduce a compact but robust NER evaluation that compares BERT (token classification) and T5 (seq2seq) across:
- **Evaluation lenses:** token macro-F1 (excluding `O`), labelled vs. unlabelled span-F1
- **Tagging variants:** full BIO and (optionally) simplified BIO
- **Domains:** in-domain (EWT) vs. out-of-domain (PUD)
- **Diagnostics:** boundary vs. type errors, entity length buckets, rarity buckets
- **Reliability:** paired bootstrap for span-F1; optional 3-seed variance

This work targets an **Eval4NLP** short paper.

---

## Data
I use **UniversalNER** English EWT and PUD IOB2 splits. The notebook downloads these automatically from public GitHub mirrors.

---

## Environment
- Python ≥ 3.10
- Recommended: NVIDIA GPU with CUDA (CPU works but is slow for T5)
- See `requirements.txt` for versions that I tested

Add this cell at the **top** of the notebook to snapshot versions:
```python
import sys, torch, transformers, pandas as pd
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda, "| is_cuda:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("Pandas:", pd.__version__)

