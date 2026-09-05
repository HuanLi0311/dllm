# Vendored SMDM subset

This directory contains the subset of
[ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM) commit
`583aa4716d17728dbb825aec6c24a121164d616a` needed to instantiate the SMDM-219M
and SMDM-1.14B checkpoints used by the geometry probes.

- `lit_gpt/` contains model definitions and their local dependencies.
- `pretrain/train_mdm.py` is retained because accepted evidence records its
  source hash as part of the training-code provenance.
- `data/gsm8k/test.jsonl` is the GSM8K source used by the LLaDA probe.
- `lit_gpt/compat.py` and small import changes in `__init__.py`, `diffmodel.py`,
  `model.py`, and `rmsnorm.py` provide native PyTorch fallbacks for optional
  fused CUDA extensions.

The upstream code is licensed under Apache License 2.0; see `LICENSE`. GSM8K
retains its own MIT notice in `data/gsm8k/LICENSE`.
