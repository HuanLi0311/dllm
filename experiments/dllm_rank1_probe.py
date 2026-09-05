#!/usr/bin/env python3
"""Measure rank-1 and diagonal Fisher surrogates for a masked DLLM."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import statistics
import sys
import traceback
from pathlib import Path


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse_floats(value):
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(not 0.0 < item < 1.0 for item in values):
        raise ValueError("mask probabilities must be comma-separated values in (0, 1)")
    return values


def _parse_ints(value):
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or values[0] < 2:
        raise ValueError("sample sizes must be comma-separated integers >= 2")
    return values


def _read_records(path, task_id=None, split="eval"):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if split and record.get("split") != split:
                continue
            if task_id is not None and int(record.get("task_id", -1)) != task_id:
                continue
            ids = record.get("input_ids")
            if not isinstance(ids, list) or not ids:
                raise ValueError("each selected record needs non-empty input_ids")
            records.append([int(token) for token in ids])
    if not records:
        raise ValueError("no records selected from data")
    return records


def _synthetic_records(count, length, vocab_size, seed):
    generator = random.Random(seed)
    return [[generator.randrange(vocab_size) for _ in range(length)] for _ in range(count)]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_dict(args):
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def _metrics(gradients, losses):
    # Keep the large parameter slice on CPU; only an S x S Gram matrix is
    # materialized. This is exact for the selected slice and avoids O(d^2)
    # memory for a 170M+ parameter model.
    import torch

    matrix = torch.stack(gradients).to(dtype=torch.float64)
    sample_count = matrix.shape[0]
    mean = matrix.mean(dim=0)
    mean_norm_sq = torch.dot(mean, mean)
    gram = (matrix @ matrix.T) / sample_count
    fisher_norm_sq = torch.sum(gram * gram)
    projection = matrix @ mean
    mu_f_mu = torch.mean(projection * projection)
    eps = torch.tensor(1e-30, dtype=torch.float64)
    coefficient = mu_f_mu / torch.clamp(mean_norm_sq * mean_norm_sq, min=eps)
    rank1_norm_sq = coefficient * coefficient * mean_norm_sq * mean_norm_sq
    rank1_error_sq = fisher_norm_sq - 2.0 * coefficient * mu_f_mu + rank1_norm_sq
    diagonal = torch.mean(matrix * matrix, dim=0)
    diagonal_error_sq = fisher_norm_sq - torch.sum(diagonal * diagonal)
    fisher_norm = torch.sqrt(torch.clamp(fisher_norm_sq, min=eps))
    rank1_error = torch.sqrt(torch.clamp(rank1_error_sq, min=0.0)) / fisher_norm
    diagonal_error = torch.sqrt(torch.clamp(diagonal_error_sq, min=0.0)) / fisher_norm
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvectors = torch.flip(eigenvectors, dims=[1])
    lambda1 = float(eigenvalues[0])
    lambda2 = float(eigenvalues[1]) if sample_count > 1 else 0.0
    oracle_top1_error = torch.sqrt(torch.clamp(fisher_norm_sq - eigenvalues[0] * eigenvalues[0], min=0.0)) / fisher_norm
    oracle_rank_errors = {}
    for rank in (1, 2, 4, 8, 16, 32, 64):
        if rank <= sample_count:
            residual = fisher_norm_sq - torch.sum(eigenvalues[:rank] * eigenvalues[:rank])
            oracle_rank_errors[str(rank)] = float(torch.sqrt(torch.clamp(residual, min=0.0)) / fisher_norm)
    # A data-span null: replace the all-positive mean with random signed means,
    # then optimally scale each resulting rank-1 direction.
    generator = torch.Generator().manual_seed(314159 + sample_count)
    signs = 2.0 * torch.randint(0, 2, (64, sample_count), generator=generator, dtype=torch.float64) - 1.0
    signed_projection = signs @ gram
    signed_norm_sq = torch.sum(signed_projection * signs, dim=1) / sample_count
    signed_fisher_quadratic = torch.sum(signed_projection * signed_projection, dim=1) / sample_count
    signed_captured_sq = (signed_fisher_quadratic / torch.clamp(signed_norm_sq, min=eps)) ** 2
    signed_errors = torch.sqrt(torch.clamp(fisher_norm_sq - signed_captured_sq, min=0.0)) / fisher_norm
    norms = torch.linalg.vector_norm(matrix, dim=1)
    cosine = (matrix @ mean) / torch.clamp(norms * torch.sqrt(mean_norm_sq), min=eps)
    trace = torch.sum(diagonal)
    return {
        "sample_count": int(sample_count),
        "slice_numel": int(matrix.shape[1]),
        "mean_gradient_norm": float(torch.sqrt(torch.clamp(mean_norm_sq, min=0.0))),
        "fisher_trace": float(trace),
        "fisher_frobenius_norm": float(fisher_norm),
        "rank1_coefficient": float(coefficient),
        # For a rank-1 matrix c*mu*mu^T, captured Fisher trace is c*||mu||^2;
        # its squared Frobenius norm is different and is used only above for
        # the reconstruction error.
        "rank1_explained_variance": float((coefficient * mean_norm_sq) / torch.clamp(trace, min=eps)),
        "rank1_relative_frobenius_error": float(rank1_error),
        "diagonal_relative_frobenius_error": float(diagonal_error),
        # Best unconstrained rank-1 error from the same sample Gram matrix.
        # This is an oracle diagnostic, not an available consolidation vector.
        "oracle_top1_relative_frobenius_error": float(oracle_top1_error),
        "oracle_rank_relative_frobenius_error": oracle_rank_errors,
        "effective_rank": float((trace * trace) / torch.clamp(fisher_norm_sq, min=eps)),
        "oracle_top1_trace_fraction": float(eigenvalues[0] / torch.clamp(trace, min=eps)),
        "mean_top1_absolute_cosine": float(
            torch.abs(torch.sqrt(eigenvalues[0] / sample_count) * torch.sum(eigenvectors[:, 0]))
            / torch.sqrt(torch.clamp(mean_norm_sq, min=eps))
        ),
        "signed_mean_null_error_mean": float(torch.mean(signed_errors)),
        "signed_mean_null_error_q05": float(torch.quantile(signed_errors, 0.05)),
        "signed_mean_null_error_q95": float(torch.quantile(signed_errors, 0.95)),
        "signed_mean_null_outperform_fraction": float(torch.mean((signed_errors <= rank1_error + 1e-12).to(torch.float64))),
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda2_over_lambda1": lambda2 / max(lambda1, 1e-30),
        "cosine_to_mean_mean": float(torch.mean(cosine)),
        "cosine_to_mean_std": float(torch.std(cosine, unbiased=False)),
        "cosine_to_mean_min": float(torch.min(cosine)),
        "absolute_cosine_to_mean_mean": float(torch.mean(torch.abs(cosine))),
        "loss_mean": float(sum(losses) / len(losses)),
        "loss_std": float(math.sqrt(sum((loss - sum(losses) / len(losses)) ** 2 for loss in losses) / len(losses))),
    }


def _split_metrics(calibration_gradients, test_gradients, calibration_losses, test_losses):
    """Fit surrogates on calibration gradients and score an independent Fisher."""
    import torch

    calibration = torch.stack(calibration_gradients).to(dtype=torch.float64)
    test = torch.stack(test_gradients).to(dtype=torch.float64)
    count, dimension = calibration.shape
    test_count = test.shape[0]
    eps = torch.tensor(1e-30, dtype=torch.float64)
    calibration_gram = (calibration @ calibration.T) / count
    test_gram = (test @ test.T) / test_count
    test_fisher_norm_sq = torch.sum(test_gram * test_gram)
    test_fisher_norm = torch.sqrt(torch.clamp(test_fisher_norm_sq, min=eps))
    calibration_mean = calibration.mean(dim=0)
    mean_norm_sq = torch.dot(calibration_mean, calibration_mean)
    mean_fisher_quadratic = torch.mean((calibration @ calibration_mean) ** 2)
    mean_coefficient = mean_fisher_quadratic / torch.clamp(mean_norm_sq * mean_norm_sq, min=eps)

    def relative_error(inner, surrogate_norm_sq):
        residual = test_fisher_norm_sq - 2.0 * inner + surrogate_norm_sq
        return float(torch.sqrt(torch.clamp(residual, min=0.0)) / test_fisher_norm)

    mean_test_quadratic = torch.mean((test @ calibration_mean) ** 2)
    mean_rank1_error = relative_error(
        mean_coefficient * mean_test_quadratic,
        mean_coefficient * mean_coefficient * mean_norm_sq * mean_norm_sq,
    )
    calibration_diagonal = torch.mean(calibration * calibration, dim=0)
    test_diagonal = torch.mean(test * test, dim=0)
    diagonal_error = relative_error(torch.dot(calibration_diagonal, test_diagonal), torch.dot(calibration_diagonal, calibration_diagonal))
    isotropic_coefficient = torch.sum(calibration_diagonal) / dimension
    isotropic_error = relative_error(
        isotropic_coefficient * torch.sum(test_diagonal),
        isotropic_coefficient * isotropic_coefficient * dimension,
    )

    calibration_eigenvalues, calibration_eigenvectors = torch.linalg.eigh(calibration_gram)
    lambda1 = torch.clamp(calibration_eigenvalues[-1], min=eps)
    top_direction = calibration.T @ calibration_eigenvectors[:, -1] / torch.sqrt(count * lambda1)
    top_error = relative_error(lambda1 * torch.mean((test @ top_direction) ** 2), lambda1 * lambda1)

    generator = torch.Generator().manual_seed(271828 + count + dimension)
    signs = 2.0 * torch.randint(0, 2, (64, count), generator=generator, dtype=torch.float64) - 1.0
    signed_directions = (signs @ calibration) / count
    signed_norm_sq = torch.sum(signed_directions * signed_directions, dim=1)
    signed_calibration_quadratic = torch.mean((calibration @ signed_directions.T) ** 2, dim=0)
    signed_coefficients = signed_calibration_quadratic / torch.clamp(signed_norm_sq * signed_norm_sq, min=eps)
    signed_test_quadratic = torch.mean((test @ signed_directions.T) ** 2, dim=0)
    signed_inner = signed_coefficients * signed_test_quadratic
    signed_surrogate_norm_sq = signed_coefficients * signed_coefficients * signed_norm_sq * signed_norm_sq
    signed_errors = torch.sqrt(torch.clamp(test_fisher_norm_sq - 2.0 * signed_inner + signed_surrogate_norm_sq, min=0.0)) / test_fisher_norm

    random_directions = torch.randn((64, dimension), generator=generator, dtype=torch.float64)
    random_directions /= torch.linalg.vector_norm(random_directions, dim=1, keepdim=True)
    random_coefficients = torch.mean((calibration @ random_directions.T) ** 2, dim=0)
    random_test_quadratic = torch.mean((test @ random_directions.T) ** 2, dim=0)
    random_errors = torch.sqrt(
        torch.clamp(test_fisher_norm_sq - 2.0 * random_coefficients * random_test_quadratic + random_coefficients * random_coefficients, min=0.0)
    ) / test_fisher_norm

    calibration_metrics = _metrics(calibration_gradients, calibration_losses)
    test_metrics = _metrics(test_gradients, test_losses)
    return {
        "sample_count": count,
        "calibration_sample_count": count,
        "test_sample_count": test_count,
        "slice_numel": dimension,
        "calibration_loss_mean": statistics.fmean(calibration_losses),
        "test_loss_mean": statistics.fmean(test_losses),
        "calibration_rank1_relative_frobenius_error": calibration_metrics["rank1_relative_frobenius_error"],
        "calibration_diagonal_relative_frobenius_error": calibration_metrics["diagonal_relative_frobenius_error"],
        "mean_rank1_test_relative_frobenius_error": mean_rank1_error,
        "diagonal_test_relative_frobenius_error": diagonal_error,
        "calibration_top1_test_relative_frobenius_error": top_error,
        "isotropic_test_relative_frobenius_error": isotropic_error,
        "test_oracle_top1_relative_frobenius_error": test_metrics["oracle_top1_relative_frobenius_error"],
        "test_effective_rank": test_metrics["effective_rank"],
        "calibration_mean_top1_absolute_cosine": calibration_metrics["mean_top1_absolute_cosine"],
        "signed_mean_test_error_mean": float(torch.mean(signed_errors)),
        "signed_mean_test_error_q05": float(torch.quantile(signed_errors, 0.05)),
        "signed_mean_test_error_q95": float(torch.quantile(signed_errors, 0.95)),
        "signed_mean_test_outperform_fraction": float(torch.mean((signed_errors <= mean_rank1_error + 1e-12).to(torch.float64))),
        "random_direction_test_error_mean": float(torch.mean(random_errors)),
        "random_direction_test_error_q05": float(torch.quantile(random_errors, 0.05)),
        "random_direction_test_error_q95": float(torch.quantile(random_errors, 0.95)),
    }


def _sample_masks(probabilities, length, device, generator):
    """Sample nonempty Bernoulli masks, resampling empty rows."""
    import torch

    masks = torch.rand((len(probabilities), length), device=device, generator=generator) < probabilities[:, None]
    empty = ~masks.any(dim=1)
    while torch.any(empty):
        masks[empty] = torch.rand((int(empty.sum()), length), device=device, generator=generator) < probabilities[empty, None]
        empty = ~masks.any(dim=1)
    return masks


def _run(args):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    code_root = Path(args.code_root).resolve()
    sys.path.insert(0, str(code_root))
    from lit_gpt.diffmodel import TransEncoder
    from lit_gpt.config import Config

    if args.model_size not in (170, 1028):
        raise ValueError("--model-size must be 170 or 1028 for supplied checkpoints")
    config = Config.from_name("Diff_LLaMA_%dM" % args.model_size)
    model = TransEncoder(config)
    state = load_file(str(Path(args.checkpoint).resolve()), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError("checkpoint keys mismatch: missing=%s unexpected=%s" % (missing, unexpected))
    del state
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    named = dict(model.named_parameters())
    parameter_names = [item.strip() for item in args.parameter.split(",") if item.strip()]
    missing_parameters = [name for name in parameter_names if name not in named]
    if missing_parameters:
        raise KeyError("parameter not found: %s" % ", ".join(missing_parameters))
    targets = [named[name] for name in parameter_names]
    for target in targets:
        if not target.requires_grad:
            target.requires_grad_(True)
    vocab_size = config.vocab_size
    sample_sizes = _parse_ints(args.sample_sizes) if args.sample_sizes else [args.samples]
    max_samples = max(sample_sizes)
    required_samples = max_samples + args.test_samples
    if args.data:
        records = _read_records(args.data, args.task_id, args.split)
        if args.shuffle_records:
            random.Random(args.seed).shuffle(records)
    else:
        records = _synthetic_records(required_samples, args.sequence_length, vocab_size, args.seed)
    records = records[:required_samples]
    if len(records) < required_samples:
        raise ValueError("requested %d examples but only %d records were selected" % (required_samples, len(records)))
    if len(records) < 2:
        raise ValueError("at least two examples are needed for lambda2")
    sequence_length = min(args.sequence_length, max(len(record) for record in records))
    records = [record[:sequence_length] for record in records]
    ids = torch.tensor(records, dtype=torch.long, device=device)
    if torch.any(ids < 0) or torch.any(ids >= vocab_size):
        raise ValueError("data token is outside model vocabulary")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    target_uniform = torch.rand(ids.shape, device=device, generator=generator)
    results = []
    conditions = [(f"fixed_{probability:g}", probability, torch.full((len(records),), probability, device=device)) for probability in _parse_floats(args.mask_probabilities)]
    if args.include_native_schedule:
        native_probabilities = (1.0 - args.native_eps) * torch.rand((len(records),), device=device, generator=generator) + args.native_eps
        conditions.append(("native_schedule", None, native_probabilities))
    for condition, probability, probabilities in conditions:
        gradients = {name: [] for name in parameter_names}
        losses = []
        masks = _sample_masks(probabilities, sequence_length, device, generator)
        for index in range(ids.shape[0]):
            clean = ids[index]
            mask = masks[index]
            noisy = clean.clone()
            noisy[mask] = vocab_size
            model.zero_grad(set_to_none=True)
            logits = model(noisy.unsqueeze(0))[0].float()
            if args.loss_mode == "mean_masked_ce":
                loss = F.cross_entropy(logits[mask], clean[mask], reduction="mean")
            else:
                target_mask = mask
                if args.loss_mode == "fixed_target":
                    target_mask = torch.zeros_like(mask)
                    target_mask[torch.argmin(target_uniform[index].masked_fill(~mask, float("inf")))] = True
                losses_by_token = F.cross_entropy(logits[target_mask], clean[target_mask], reduction="none")
                loss = losses_by_token.sum() / (probabilities[index] * sequence_length)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss in mask condition %s" % condition)
            example_gradients = torch.autograd.grad(loss, targets, retain_graph=False, create_graph=False, allow_unused=False)
            if any(not torch.isfinite(gradient).all() for gradient in example_gradients):
                raise FloatingPointError("non-finite gradient in mask condition %s" % condition)
            for name, gradient in zip(parameter_names, example_gradients):
                gradients[name].append(gradient.detach().float().cpu().reshape(-1))
            losses.append(float(loss.detach().cpu()))
            del logits, loss, example_gradients
        for name in parameter_names:
            for sample_count in sample_sizes:
                if args.test_samples:
                    metric = _split_metrics(
                        gradients[name][:sample_count],
                        gradients[name][max_samples:],
                        losses[:sample_count],
                        losses[max_samples:],
                    )
                else:
                    metric = _metrics(gradients[name][:sample_count], losses[:sample_count])
                metric.update(
                    {
                        "model_size_m": args.model_size,
                        "mask_probability": probability,
                        "mask_probability_mean": float(torch.mean(probabilities)),
                        "mask_condition": condition,
                        "mask_analogue": "higher_mask_probability=operationally_higher_noise; not Gaussian SNR",
                        "loss_mode": args.loss_mode,
                        "evaluation": "split_sample" if args.test_samples else "in_sample",
                        "parameter": name,
                        "seed": args.seed,
                        "sequence_length": sequence_length,
                    }
                )
                results.append(metric)
    return {
        "schema_version": 1,
        "status": "ok",
        "experiment": "dllm_rank1_fisher",
        "claim": "split-sample rank-1 Fisher geometry under masked diffusion",
        "host": os.uname().nodename,
        "created_utc": _utc_now(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "code_root": str(code_root),
        "probe_sha256": _sha256(__file__),
        "source_sha256": {
            "lit_gpt/diffmodel.py": _sha256(code_root / "lit_gpt/diffmodel.py"),
            "lit_gpt/config.py": _sha256(code_root / "lit_gpt/config.py"),
            "pretrain/train_mdm.py": _sha256(code_root / "pretrain/train_mdm.py"),
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        },
        "command": [sys.executable, *sys.argv],
        "data": str(Path(args.data).resolve()) if args.data else "synthetic",
        "data_sha256": _sha256(args.data) if args.data else None,
        "data_split": args.split if args.data else None,
        "task_id": args.task_id,
        "config": _config_dict(args),
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-size", type=int)
    parser.add_argument("--code-root")
    parser.add_argument("--data")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--mask-probabilities", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sample-sizes", help="reuse one gradient batch for comma-separated prefix sizes")
    parser.add_argument("--test-samples", type=int, default=0, help="held-out gradients used only to score fitted surrogates")
    parser.add_argument("--shuffle-records", action="store_true", help="seeded shuffle before selecting data records")
    parser.add_argument("--loss-mode", choices=("mean_masked_ce", "native_conditional", "fixed_target"), default="mean_masked_ce")
    parser.add_argument("--include-native-schedule", action="store_true")
    parser.add_argument("--native-eps", type=float, default=1e-3)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--parameter", default="transformer.h.0.attn.proj.weight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        import torch

        assert _parse_floats("0.1,0.9") == [0.1, 0.9]
        assert _parse_ints("8,4,8") == [4, 8]
        stats = _metrics(
            [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
            [1.0, 1.0, 1.0],
        )
        assert stats["oracle_top1_relative_frobenius_error"] <= stats["rank1_relative_frobenius_error"] + 1e-12
        assert 0.0 <= stats["mean_top1_absolute_cosine"] <= 1.0 + 1e-12
        assert set(stats["oracle_rank_relative_frobenius_error"]) == {"1", "2"}
        calibration = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0]), torch.tensor([1.0, 1.0])]
        test = [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, -1.0])]
        split = _split_metrics(calibration, test, [1.0] * 3, [1.0] * 3)
        calibration_matrix = torch.stack(calibration).double()
        test_matrix = torch.stack(test).double()
        calibration_fisher = calibration_matrix.T @ calibration_matrix / 3
        test_fisher = test_matrix.T @ test_matrix / 3
        mean = calibration_matrix.mean(dim=0)
        coefficient = (mean @ calibration_fisher @ mean) / (mean @ mean) ** 2
        direct_rank1_error = torch.linalg.matrix_norm(test_fisher - coefficient * torch.outer(mean, mean)) / torch.linalg.matrix_norm(test_fisher)
        direct_diagonal_error = torch.linalg.matrix_norm(test_fisher - torch.diag(torch.diag(calibration_fisher))) / torch.linalg.matrix_norm(test_fisher)
        assert math.isclose(split["mean_rank1_test_relative_frobenius_error"], float(direct_rank1_error), rel_tol=1e-12)
        assert math.isclose(split["diagonal_test_relative_frobenius_error"], float(direct_diagonal_error), rel_tol=1e-12)
        assert split["test_oracle_top1_relative_frobenius_error"] <= split["mean_rank1_test_relative_frobenius_error"] + 1e-12
        print(json.dumps({"self_check": "ok"}))
        return 0
    for required in ("checkpoint", "model_size", "code_root", "output"):
        if getattr(args, required) is None:
            parser.error("--%s is required unless --self-check is used" % required.replace("_", "-"))
    if args.test_samples == 1 or args.test_samples < 0:
        parser.error("--test-samples must be zero or at least two")
    if not 0.0 < args.native_eps < 1.0:
        parser.error("--native-eps must be in (0, 1)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = _run(args)
        exit_code = 0
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "experiment": "dllm_rank1_fisher",
            "created_utc": _utc_now(),
            "config": _config_dict(args),
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        }
        exit_code = 2
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
