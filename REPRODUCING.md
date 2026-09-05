# Reproducing the audit

Run every command from the repository root. The published evidence can be
verified on CPU; rerunning model probes requires an NVIDIA GPU and downloaded
checkpoints.

## Environments

The SMDM runs used Python 3.9, PyTorch 2.4.1+cu121, and an A100 40GB:

```bash
python -m venv .venv-smdm
source .venv-smdm/bin/activate
python -m pip install -r requirements.txt
```

LLaDA used Python 3.10, PyTorch 2.5.1+cu121, and Transformers 4.46.0. Install it
in a separate environment:

```bash
python -m venv .venv-llada
source .venv-llada/bin/activate
python -m pip install -r requirements-llada.txt
```

## Checkpoints

Weights are not included. With `huggingface-hub` installed:

```bash
hf download nieshen/SMDM \
  mdm_safetensors/mdm-170M-100e18.safetensors \
  mdm_safetensors/mdm-1028M-1600e18.safetensors \
  --local-dir checkpoints

hf download GSAI-ML/LLaDA-8B-Base \
  --local-dir checkpoints/llada-8b-base
```

Expected SHA-256 values are:

- SMDM-219M: `2d8c9b9a730715f2c772d5bc740e12951fc160e5e8511a16835f3537401ea9bb`
- SMDM-1.14B: `ce96ce67a051613b6d7feb419c99c0b4db5bfcfaaa0833ed7f7ecbc6632841d6`
- LLaDA-8B aggregate: `b84552bd96af3dc51fb9782085672269e95c1e4fe1eebd2a901863a8739a1b95`

The LLaDA aggregate hashes the model index, configuration, remote model code,
and all weight shards; individual hashes are stored in each public envelope.

## Probe examples

The checked-in SMDM input is `runs/data/gsm8k_tasks.jsonl`. A representative
SMDM-219M seed is:

```bash
python experiments/dllm_rank1_probe.py \
  --checkpoint checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors \
  --model-size 170 \
  --code-root SMDM \
  --data runs/data/gsm8k_tasks.jsonl \
  --split eval \
  --mask-probabilities 0.1,0.3,0.5,0.7,0.9 \
  --sample-sizes 32,64,128 \
  --test-samples 128 \
  --shuffle-records \
  --loss-mode native_conditional \
  --include-native-schedule \
  --sequence-length 64 \
  --parameter transformer.h.0.norm_1.weight,transformer.h.8.norm_1.weight,transformer.h.17.norm_1.weight \
  --seed 0 \
  --device cuda \
  --output runs/local/smdm-219m-seed0.json
```

The LLaDA probe tokenizes source documents independently:

```bash
python experiments/llada_geometry_probe.py \
  --checkpoint checkpoints/llada-8b-base \
  --data SMDM/data/gsm8k/test.jsonl \
  --mask-probabilities 0.1,0.3,0.5,0.7,0.9 \
  --sample-sizes 16,32,64 \
  --test-samples 64 \
  --sequence-length 64 \
  --parameter model.transformer.blocks.0.attn_norm.weight,model.transformer.blocks.15.attn_norm.weight,model.transformer.blocks.31.attn_norm.weight \
  --seed 0 \
  --device cuda \
  --output runs/local/llada-8b-seed0.json
```

Every successful envelope records its full command, software versions, source
hashes, checkpoint hashes, selected data, configuration, and direct metrics.

## Null and figures

Regenerate the CPU null:

```bash
python experiments/simulate_fisher_null.py \
  --case 768:8:128 --case 768:16:128 --case 768:32:128 \
  --case 768:64:128 --case 768:128:128 \
  --case 1792:8:64 --case 1792:16:64 --case 1792:32:64 \
  --case 1792:64:64 \
  --repetitions 200 \
  --seed 20260825 \
  --device cpu \
  --output runs/local/isotropic_null.json
```

Then regenerate all plots directly from the compressed public evidence:

```bash
python experiments/make_paper_figures.py \
  --geometry \
    evidence/raw/runs/r08_split_primary/gsm_170_s*/benchmark.json.gz \
    evidence/raw/runs/r08_split_primary/gsm_1028_s*/benchmark.json.gz \
    evidence/raw/runs/r16_llada_geometry/*.json.gz \
  --comparison-control \
    evidence/raw/runs/r12_audited_controls/gsm_all_170_s*/benchmark.json.gz \
    evidence/raw/runs/r12_audited_controls/gsm_one_170_s*/benchmark.json.gz \
  --comparison-contract evidence/comparison_contract.json \
  --slice-control \
    evidence/raw/runs/r10_attention_split/*/benchmark.json.gz \
    evidence/raw/runs/r17_late_dense_matched/*.json.gz \
  --null runs/local/isotropic_null.json \
  --output-dir figures
```

`figures/figure_data.json` records the hash of every consumed input. Generated
figures are ignored by Git because they are reproducible products.

## Public evidence verification

```bash
python experiments/build_review_bundle.py --verify evidence/release_manifest.json
python experiments/build_comparison_contract.py --self-check
```

The first command verifies all 25 compressed hashes and scans decompressed JSON
for identity-bearing absolute paths. The checked-in comparison contract also
recomputes rank-1 and diagonal errors from stored sufficient statistics when
the figure script loads it.
