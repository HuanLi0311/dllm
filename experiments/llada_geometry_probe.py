#!/usr/bin/env python3
"""Split-sample Fisher geometry probe for the public LLaDA-8B base model."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import random
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).parents[1]
BASE_PROBE = Path(__file__).with_name("dllm_rank1_probe.py")


def _base_probe():
    spec = importlib.util.spec_from_file_location("dllm_rank1_probe", BASE_PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path, tokenizer, sequence_length, seed, count):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record["question"].strip() + "\n" + record["answer"].strip()
            tokens = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
            if len(tokens) >= sequence_length:
                rows.append((source_line, tokens[:sequence_length]))
    random.Random(seed).shuffle(rows)
    if len(rows) < count:
        raise ValueError(f"need {count} full-length documents, found {len(rows)}")
    return rows[:count]


def _checkpoint_hashes(checkpoint):
    files = [checkpoint / "config.json", checkpoint / "model.safetensors.index.json", checkpoint / "modeling_llada.py"]
    files.extend(sorted(checkpoint.glob("model-*.safetensors")))
    return {path.name: _sha256(path) for path in files}


def _run(args):
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModel, AutoTokenizer

    base = _base_probe()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint = args.checkpoint.resolve()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    sample_sizes = base._parse_ints(args.sample_sizes)
    maximum = max(sample_sizes)
    selected = _records(args.data, tokenizer, args.sequence_length, args.seed, maximum + args.test_samples)
    ids = torch.tensor([tokens for _, tokens in selected], dtype=torch.long, device=device)

    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named = dict(model.named_parameters())
    parameter_names = [name.strip() for name in args.parameter.split(",") if name.strip()]
    missing = [name for name in parameter_names if name not in named]
    if missing:
        raise KeyError("parameter not found: " + ", ".join(missing))
    targets = [named[name].requires_grad_(True) for name in parameter_names]

    generator = torch.Generator(device=device).manual_seed(args.seed)
    results = []
    probabilities_to_run = base._parse_floats(args.mask_probabilities)
    for probability in probabilities_to_run:
        probabilities = torch.full((len(selected),), probability, device=device)
        masks = base._sample_masks(probabilities, args.sequence_length, device, generator)
        gradients = {name: [] for name in parameter_names}
        losses = []
        for index, clean in enumerate(ids):
            noisy = clean.clone()
            noisy[masks[index]] = model.config.mask_token_id
            logits = model(noisy.unsqueeze(0)).logits[0].float()
            token_losses = F.cross_entropy(logits[masks[index]], clean[masks[index]], reduction="none")
            loss = token_losses.sum() / (probability * args.sequence_length)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at p={probability:g}")
            example_gradients = torch.autograd.grad(loss, targets, allow_unused=False)
            if any(not torch.isfinite(gradient).all() for gradient in example_gradients):
                raise FloatingPointError(f"non-finite gradient at p={probability:g}")
            for name, gradient in zip(parameter_names, example_gradients):
                gradients[name].append(gradient.detach().float().cpu().reshape(-1))
            losses.append(float(loss.detach().cpu()))
            del logits, loss, example_gradients
        for name in parameter_names:
            for sample_count in sample_sizes:
                metric = base._split_metrics(
                    gradients[name][:sample_count],
                    gradients[name][maximum:],
                    losses[:sample_count],
                    losses[maximum:],
                )
                metric.update({
                    "model": "LLaDA-8B-Base",
                    "model_size_m": 8016,
                    "mask_probability": probability,
                    "mask_probability_mean": probability,
                    "mask_condition": f"fixed_{probability:g}",
                    "mask_analogue": "mask probability is not Gaussian SNR",
                    "loss_mode": "native_conditional",
                    "evaluation": "split_sample",
                    "parameter": name,
                    "seed": args.seed,
                    "sequence_length": args.sequence_length,
                })
                results.append(metric)

    checkpoint_hashes = _checkpoint_hashes(checkpoint)
    return {
        "schema_version": 1,
        "status": "ok",
        "experiment": "llada_rank1_fisher",
        "claim": "split-sample rank-1 Fisher geometry in LLaDA-8B-Base",
        "host": os.uname().nodename,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(json.dumps(checkpoint_hashes, sort_keys=True).encode()).hexdigest(),
        "checkpoint_files_sha256": checkpoint_hashes,
        "probe_sha256": _sha256(__file__),
        "base_probe_sha256": _sha256(BASE_PROBE),
        "source_sha256": {
            "config.json": checkpoint_hashes["config.json"],
            "modeling_llada.py": checkpoint_hashes["modeling_llada.py"],
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "command": [sys.executable, *sys.argv],
        "data": str(args.data.resolve()),
        "data_sha256": _sha256(args.data),
        "data_split": "official GSM8K test documents",
        "selected_source_lines": [source_line for source_line, _ in selected],
        "config": {
            "checkpoint": str(args.checkpoint),
            "data": str(args.data),
            "mask_probabilities": args.mask_probabilities,
            "sample_sizes": args.sample_sizes,
            "test_samples": args.test_samples,
            "sequence_length": args.sequence_length,
            "parameter": args.parameter,
            "seed": args.seed,
            "device": args.device,
        },
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--mask-probabilities", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--sample-sizes", default="16,32,64")
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--parameter", default="model.transformer.blocks.0.attn_norm.weight,model.transformer.blocks.15.attn_norm.weight,model.transformer.blocks.31.attn_norm.weight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        import torch

        base = _base_probe()
        metrics = base._split_metrics(
            [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
            [torch.tensor([1.0, 1.0]), torch.tensor([1.0, -1.0])],
            [1.0, 1.0],
            [1.0, 1.0],
        )
        assert metrics["test_oracle_top1_relative_frobenius_error"] <= metrics["mean_rank1_test_relative_frobenius_error"]
        print(json.dumps({"self_check": "ok"}))
        return 0
    if not args.checkpoint or not args.data or not args.output:
        parser.error("--checkpoint, --data, and --output are required")
    if args.test_samples < 2:
        parser.error("--test-samples must be at least two")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = _run(args)
        status = 0
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        }
        status = 2
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
