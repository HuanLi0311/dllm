# Split-sample rank-1 Fisher audit

Code and public evidence for **A Split-Sample Audit of Rank-1 Fisher Geometry
in Masked Diffusion Language Models**.

The project tests whether a mean-gradient rank-1 surrogate is a reliable
approximation to empirical Fisher geometry when fitting and evaluation use
disjoint gradient samples. It measures direct relative Frobenius error on
selected SMDM-219M, SMDM-1.14B, and LLaDA-8B parameter slices. GSM8K supplies
standardized text only; this is not a task-accuracy or downstream EWC study.

## Quick start

The small checks run without checkpoints or GPUs. The SMDM environment used
Python 3.9 and CUDA 12.1:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

for script in \
  simulate_fisher_null.py \
  dllm_rank1_probe.py \
  llada_geometry_probe.py \
  run_audited_geometry_probe.py \
  make_paper_figures.py \
  build_comparison_contract.py \
  build_submission_manifest.py \
  build_review_bundle.py
do
  PYTHONNOUSERSITE=1 python "experiments/$script" --self-check
done
```

LLaDA-8B needs a separate Python 3.10 environment because its model code uses a
newer Transformers release:

```bash
python -m pip install -r requirements-llada.txt
```

Checkpoints are intentionally excluded. Download instructions, complete probe
commands, and figure regeneration are in [REPRODUCING.md](REPRODUCING.md).

## Public evidence

`evidence/raw/` contains the 25 accepted experiment envelopes as deterministic,
identity-scrubbed JSON gzip files. No file exceeds GitHub's 100MB file limit.
Verify every compressed-file hash and scan the decompressed content with:

```bash
python experiments/build_review_bundle.py --verify evidence/release_manifest.json
```

Local `runs/` outputs record exact machines and paths and are therefore ignored
by Git except for the deterministic input data. Rebuild `evidence/` before a
release; never publish the local envelopes directly.

## Layout

- `experiments/` — probes, controls, plots, and fail-closed verifiers.
- `evidence/` — public compressed envelopes and their hash manifests.
- `runs/data/` — deterministic token IDs used by the SMDM probes.
- `SMDM/` — the Apache-2.0 upstream model code needed to load SMDM checkpoints.
- `report/INVALIDATED_RESULTS.md` — reasons superseded pilots are excluded.

## License

Original code in this repository is MIT licensed. Vendored code, data, and
downloaded checkpoints retain their upstream licenses; see
[NOTICE.md](NOTICE.md).
