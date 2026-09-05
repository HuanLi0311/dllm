# Third-party notices

This repository contains or interoperates with the following third-party
materials. They are not relicensed by the repository-level MIT license.

## SMDM

`SMDM/` is a reduced snapshot of the official
[ML-GSAI/SMDM](https://github.com/ML-GSAI/SMDM) implementation at commit
`583aa4716d17728dbb825aec6c24a121164d616a`. It retains its Apache License 2.0
in `SMDM/LICENSE`. The snapshot keeps the model-loading code and the original
`pretrain/train_mdm.py` used for provenance; unrelated training, evaluation,
and benchmark assets are omitted. Local changes in `lit_gpt/__init__.py`,
`diffmodel.py`, `model.py`, and `rmsnorm.py`, plus the new `compat.py`, replace
hard dependencies on optional fused CUDA extensions with native PyTorch
fallbacks.

## GSM8K

`SMDM/data/gsm8k/test.jsonl` and the derived token IDs in `runs/data/` come from
OpenAI's [GSM8K repository](https://github.com/openai/grade-school-math), which
is MIT licensed. Its license notice is retained in `SMDM/data/gsm8k/LICENSE`.

## Model checkpoints

Model weights are not redistributed. SMDM checkpoints are downloaded from
`nieshen/SMDM` and retain that project's terms. LLaDA-8B-Base is downloaded
from [GSAI-ML/LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
and is published under the MIT license. Review the upstream model cards before
redistributing weights or derived models.
