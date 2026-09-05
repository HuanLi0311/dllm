"""Streaming Fisher diagnostics for a cached SMDM A-stage checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from iclr_1.continual_benchmark import REVERSE_DIR, encode_benchmark_rows, select
from iclr_1.continual_mdm import collate, load_cache, load_model, per_example_loss, set_seed, trainable_parameters
from iclr_1.continual_reverse import fact_rows


ROOT = Path(__file__).resolve().parent


def vector_norm(values: np.ndarray, chunk_size: int = 16_777_216) -> float:
    total = 0.0
    for start in range(0, values.size, chunk_size):
        block = np.asarray(values[start : start + chunk_size], dtype=np.float64)
        total += float(np.dot(block, block))
    return total ** 0.5


def vector_dot(left: np.ndarray, right: np.ndarray, chunk_size: int = 16_777_216) -> float:
    total = 0.0
    for start in range(0, left.size, chunk_size):
        left_block = np.asarray(left[start : start + chunk_size], dtype=np.float64)
        right_block = np.asarray(right[start : start + chunk_size], dtype=np.float64)
        total += float(np.dot(left_block, right_block))
    return total


def bounded_cosine(left: np.ndarray, right: np.ndarray, chunk_size: int = 16_777_216) -> float:
    denominator = max(vector_norm(left, chunk_size) * vector_norm(right, chunk_size), 1e-24)
    return min(1.0, max(0.0, abs(vector_dot(left, right, chunk_size)) / denominator))


def gradient_block(gradients: np.ndarray, start: int, stop: int, scales: np.ndarray | None) -> np.ndarray:
    block = np.asarray(gradients[:, start:stop])
    if scales is not None:
        block = block.astype(np.float32) * np.asarray(scales, dtype=np.float32)[:, None]
    return block


def summarize(gradients: np.ndarray, rank: int = 1) -> tuple[dict, np.ndarray, np.ndarray]:
    if not np.isfinite(gradients).all():
        raise FloatingPointError("gradient array contains non-finite values")
    count = gradients.shape[0]
    if not 1 <= rank <= count:
        raise ValueError(f"rank must be in [1, {count}]")
    gram = gradients @ gradients.T / count
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    full_frobenius_sq = float(np.square(eigenvalues).sum())
    full_frobenius = max(full_frobenius_sq, 1e-24) ** 0.5
    norms = np.linalg.norm(gradients, axis=1)
    safe_norms = np.maximum(norms, 1e-12)
    cosine = gram * count / (safe_norms[:, None] * safe_norms[None, :])
    pairwise = cosine[~np.eye(count, dtype=bool)]
    mean_gradient = gradients.mean(axis=0)
    mean_norm = vector_norm(mean_gradient)
    mean_direction = mean_gradient / max(mean_norm, 1e-12)
    mean_scores = gram @ np.ones(count)
    mean_norm_sq = float(mean_scores.sum() / count)
    lambda_mean = float(np.square(mean_scores).mean() / max(mean_norm_sq, 1e-24))
    top_directions = (gradients.T @ eigenvectors[:, :rank]).T
    for index in range(rank):
        top_directions[index] /= max(vector_norm(top_directions[index]), 1e-12)
    diagonal = np.square(gradients).mean(axis=0)
    diagonal_frobenius_sq = float(np.square(diagonal).sum())
    stats = {
        "fisher_examples": count,
        "fisher_dim": gradients.shape[1],
        "fisher_trace": float(np.trace(gram)),
        "fisher_lambda1": float(eigenvalues[0]),
        "fisher_lambda2": float(eigenvalues[1]) if count > 1 else 0.0,
        "fisher_lambda_mean": lambda_mean,
        "lambda1_over_trace": float(eigenvalues[0] / max(np.trace(gram), 1e-12)),
        "lambda2_over_lambda1": float(eigenvalues[1] / max(eigenvalues[0], 1e-12)) if count > 1 else 0.0,
        "lambda_mean_over_trace": float(lambda_mean / max(np.trace(gram), 1e-12)),
        "rank1_relative_frobenius_error": max(full_frobenius_sq - eigenvalues[0] ** 2, 0.0) ** 0.5 / full_frobenius,
        "mean_rank1_relative_frobenius_error": max(full_frobenius_sq - lambda_mean ** 2, 0.0) ** 0.5 / full_frobenius,
        "diagonal_relative_frobenius_error": max(full_frobenius_sq - diagonal_frobenius_sq, 0.0) ** 0.5 / full_frobenius,
        "gradient_norm_min": float(norms.min()),
        "gradient_norm_mean": float(norms.mean()),
        "gradient_norm_max": float(norms.max()),
        "gradient_zero_count": int(np.count_nonzero(norms == 0.0)),
        "gradient_pairwise_cosine": float(pairwise.mean()) if len(pairwise) else 0.0,
        "gradient_pairwise_cosine_abs": float(np.abs(pairwise).mean()) if len(pairwise) else 0.0,
        "gradient_mean_cosine": float((mean_scores / max(mean_norm, 1e-12) / safe_norms).mean()),
        "gradient_mean_cosine_abs": float(np.abs(mean_scores / max(mean_norm, 1e-12) / safe_norms).mean()),
        "top_mean_direction_cosine_abs": bounded_cosine(top_directions[0], mean_direction),
        "fisher_spectrum": [float(value) for value in eigenvalues],
    }
    return stats, top_directions, mean_direction


def summarize_memmap(
    gradients: np.memmap,
    chunk_size: int = 16_777_216,
    device: torch.device | None = None,
    whole_gradient: bool = False,
    scales: np.ndarray | None = None,
    rank: int = 1,
    workspace: Path | None = None,
    directions_output: Path | None = None,
    store_directions: bool = True,
    direction_dtype: str = "f32",
) -> tuple[dict, np.ndarray | None, np.ndarray]:
    use_gpu = device is not None and device.type == "cuda"
    if whole_gradient:
        dense = np.array(gradients, copy=True)
        if scales is not None:
            dense = dense.astype(np.float32) * np.asarray(scales, dtype=np.float32)[:, None]
        return summarize(dense, rank)
    count, dimension = gradients.shape
    gram_unscaled = np.zeros((count, count), dtype=np.float64)
    if workspace is None:
        mean_gradient = np.empty(dimension, dtype=np.float32)
        diagonal = np.empty(dimension, dtype=np.float32)
    else:
        workspace.mkdir(parents=True, exist_ok=True)
        mean_gradient = np.memmap(
            workspace / "mean_gradient.f32", mode="w+", dtype=np.float32, shape=(dimension,)
        )
        diagonal = np.memmap(
            workspace / "diagonal.f32", mode="w+", dtype=np.float32, shape=(dimension,)
        )
    for start in range(0, dimension, chunk_size):
        stop = min(start + chunk_size, dimension)
        block = gradient_block(gradients, start, stop, scales)
        if not np.isfinite(block).all():
            raise FloatingPointError(f"gradient memmap contains non-finite values in [{start}:{stop}]")
        if use_gpu:
            block_tensor = torch.from_numpy(block).to(device)
            gram_unscaled += (block_tensor @ block_tensor.T).cpu().numpy()
            mean_gradient[start:stop] = block_tensor.mean(axis=0).cpu().numpy()
            diagonal[start:stop] = block_tensor.square().mean(axis=0).cpu().numpy()
        else:
            gram_unscaled += block @ block.T
            mean_gradient[start:stop] = block.mean(axis=0)
            diagonal[start:stop] = np.square(block).mean(axis=0)
    gram = gram_unscaled / count
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    trace = float(np.trace(gram))
    full_frobenius_sq = float(np.square(eigenvalues).sum())
    full_frobenius = max(full_frobenius_sq, 1e-24) ** 0.5
    norms = np.sqrt(np.maximum(np.diag(gram) * count, 0.0))
    safe_norms = np.maximum(norms, 1e-12)
    cosine = gram * count / (safe_norms[:, None] * safe_norms[None, :])
    pairwise = cosine[~np.eye(count, dtype=bool)]
    mean_norm = vector_norm(mean_gradient, chunk_size)
    mean_gradient /= max(mean_norm, 1e-12)
    mean_direction = mean_gradient
    if not 1 <= rank <= count:
        raise ValueError(f"rank must be in [1, {count}]")
    if store_directions:
        if directions_output is None:
            top_directions = np.empty((rank, dimension), dtype=np.float32)
        else:
            directions_output.parent.mkdir(parents=True, exist_ok=True)
            top_directions = np.lib.format.open_memmap(
                directions_output,
                mode="w+",
                dtype=np.float16 if direction_dtype == "f16" else np.float32,
                shape=(rank, dimension),
            )
    else:
        top_directions = None
    direction_count = rank if store_directions else 1
    direction_norms = np.maximum((count * eigenvalues[:direction_count]) ** 0.5, 1e-12).astype(np.float32)
    top_mean_dot = 0.0
    for start in range(0, dimension, chunk_size):
        stop = min(start + chunk_size, dimension)
        block = gradient_block(gradients, start, stop, scales)
        if use_gpu:
            block_tensor = torch.from_numpy(block).to(device)
            projected = (
                block_tensor.T
                @ torch.from_numpy(eigenvectors[:, :direction_count].astype(np.float32)).to(device)
            ).cpu().numpy().T
        else:
            projected = (block.T @ eigenvectors[:, :direction_count]).T
        projected /= direction_norms[:, None]
        if top_directions is not None:
            top_directions[:, start:stop] = projected
        top_mean_dot += float(
            np.sum(projected[0] * mean_gradient[start:stop], dtype=np.float64)
        )
    mean_scores = gram @ np.ones(count)
    mean_norm_sq = float(mean_scores.sum() / count)
    lambda_mean = float(np.square(mean_scores).mean() / max(mean_norm_sq, 1e-24))
    diagonal_frobenius_sq = float(np.square(diagonal).sum())
    stats = {
        "fisher_examples": count,
        "fisher_dim": dimension,
        "fisher_trace": trace,
        "fisher_lambda1": float(eigenvalues[0]),
        "fisher_lambda2": float(eigenvalues[1]) if count > 1 else 0.0,
        "fisher_lambda_mean": lambda_mean,
        "lambda1_over_trace": float(eigenvalues[0] / max(trace, 1e-12)),
        "lambda2_over_lambda1": float(eigenvalues[1] / max(eigenvalues[0], 1e-12)) if count > 1 else 0.0,
        "lambda_mean_over_trace": float(lambda_mean / max(trace, 1e-12)),
        "rank1_relative_frobenius_error": max(full_frobenius_sq - eigenvalues[0] ** 2, 0.0) ** 0.5 / full_frobenius,
        "mean_rank1_relative_frobenius_error": max(full_frobenius_sq - lambda_mean ** 2, 0.0) ** 0.5 / full_frobenius,
        "diagonal_relative_frobenius_error": max(full_frobenius_sq - diagonal_frobenius_sq, 0.0) ** 0.5 / full_frobenius,
        "gradient_norm_min": float(norms.min()),
        "gradient_norm_mean": float(norms.mean()),
        "gradient_norm_max": float(norms.max()),
        "gradient_zero_count": int(np.count_nonzero(norms == 0.0)),
        "gradient_pairwise_cosine": float(pairwise.mean()) if len(pairwise) else 0.0,
        "gradient_pairwise_cosine_abs": float(np.abs(pairwise).mean()) if len(pairwise) else 0.0,
        "gradient_mean_cosine": float((mean_scores / max(mean_norm, 1e-12) / safe_norms).mean()),
        "gradient_mean_cosine_abs": float(np.abs(mean_scores / max(mean_norm, 1e-12) / safe_norms).mean()),
        "top_mean_direction_cosine_abs": min(1.0, abs(top_mean_dot)),
        "fisher_spectrum": [float(value) for value in eigenvalues],
    }
    del mean_gradient, diagonal
    return stats, top_directions, mean_direction


def estimate(args) -> dict:
    if args.a_cache is None or args.output is None:
        raise ValueError("--a-cache and --output are required unless --self-check is used")
    if args.examples < 2:
        raise ValueError("--examples must be at least 2")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_float32_matmul_precision("high")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    raw_rows = fact_rows(args.reverse_dir, args.task, "train", args.group_start, args.group_count)
    row_reuse = args.examples > len(raw_rows)
    if args.row_seed is None:
        selected_raw = (raw_rows * ((args.examples + len(raw_rows) - 1) // len(raw_rows)))[: args.examples]
    elif row_reuse:
        # ponytail: validation A has 60 rows; cycle rows for >60 mask draws.
        selected_raw = random.Random(args.row_seed).choices(raw_rows, k=args.examples)
    else:
        selected_raw = select(raw_rows, args.examples, args.row_seed)
    rows = encode_benchmark_rows(selected_raw, tokenizer, args.max_length)
    if len(rows) != len(selected_raw):
        raise RuntimeError("tokenizer removed Fisher rows; use a larger --max-length")

    model = load_model(args, device)
    parameters = trainable_parameters(model, args.trainable)
    payload = load_cache(args.a_cache, device)
    model.load_state_dict(payload["state_dict"])
    del payload
    model.train()

    dimension = sum(parameter.numel() for parameter in parameters)
    scratch_dir = args.scratch_dir
    scratch_dir.mkdir(parents=True, exist_ok=True)
    gradient_path = scratch_dir / f"{args.output.stem}.gradients.{args.gradient_dtype}"
    scales = None
    scale_path = None
    if args.gradient_dtype == "f16":
        scale_path = scratch_dir / f"{args.output.stem}.scales.f32"
        scales = np.memmap(scale_path, mode="w+", dtype=np.float32, shape=(len(rows),))
    # ponytail: disk-backed rows bound RAM; use randomized sketches only if disk I/O is limiting.
    storage_dtype = np.float16 if args.gradient_dtype == "f16" else np.float32
    gradients = np.memmap(gradient_path, mode="w+", dtype=storage_dtype, shape=(len(rows), dimension))
    generator = torch.Generator(device=device).manual_seed(args.fisher_seed)
    gradient_loss_values = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        ids, valid, spans = collate(batch_rows, pad_id)
        ids, valid = ids.to(device), valid.to(device)
        losses = per_example_loss(
            model,
            ids,
            valid,
            generator,
            device.type == "cuda",
            args.mask_min,
            args.mask_max,
            spans,
            args.answer_only,
            args.normalization,
            args.force_mask,
        )
        for offset, loss in enumerate(losses):
            index = start + offset
            gradient_loss_values.append(float(loss.detach().cpu()))
            grads = torch.autograd.grad(
                loss, parameters, retain_graph=offset + 1 < len(losses), allow_unused=True
            )
            vector = torch.cat([
                (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1).detach().float().cpu()
                for parameter, gradient in zip(parameters, grads)
            ])
            if not torch.isfinite(vector).all():
                raise FloatingPointError(f"non-finite Fisher gradient at example {index}")
            if scales is None:
                gradients[index] = vector.numpy()
            else:
                scale = float(vector.abs().max())
                if scale == 0.0:
                    scale = 1.0
                scales[index] = scale
                gradients[index] = (vector / scale).numpy()
            del grads, vector, loss
        model.zero_grad(set_to_none=True)
        del losses, ids, valid
    gradients.flush()
    if scales is not None:
        scales.flush()

    print("fisher_gradients_done", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats, top_directions, mean_direction = summarize_memmap(
        gradients,
        args.chunk_size,
        device,
        args.whole_gradient,
        scales,
        args.rank,
        args.summary_workspace or args.output.parent / f".{args.output.stem}_summary_workspace",
        args.directions_output,
        args.direction_output is not None or args.directions_output is not None,
        args.direction_dtype,
    )
    print("fisher_summary_done", flush=True)
    stats.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "a_cache": str(args.a_cache.resolve()),
            "task": args.task,
            "group_start": args.group_start,
            "group_count": args.group_count,
            "item_ids": [row.get("item_id") for row in selected_raw],
            "seed": args.seed,
            "fisher_seed": args.fisher_seed,
            "row_seed": args.row_seed,
            "mask_min": args.mask_min,
            "mask_max": args.mask_max,
            "force_mask": args.force_mask,
            "normalization": args.normalization,
            "answer_only": args.answer_only,
            "row_reuse": row_reuse,
            "gradient_storage": args.gradient_dtype,
            "gradient_loss_min": min(gradient_loss_values),
            "gradient_loss_mean": sum(gradient_loss_values) / len(gradient_loss_values),
            "gradient_loss_max": max(gradient_loss_values),
        }
    )
    if scales is not None:
        stats.update({
            "gradient_scale_min": float(scales.min()),
            "gradient_scale_mean": float(scales.mean()),
            "gradient_scale_max": float(scales.max()),
        })
    args.output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    if args.direction_output:
        args.direction_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.direction_output, top_directions[0].astype(np.float32))
    if args.directions_output:
        top_directions.flush()
    if args.mean_direction_output:
        args.mean_direction_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.mean_direction_output, mean_direction.astype(np.float32))
    del gradients
    if scales is not None:
        del scales
    gc.collect()
    if not args.keep_gradients:
        gradient_path.unlink()
        if scale_path is not None:
            scale_path.unlink()
    return stats


def summarize_existing(args) -> dict:
    if args.gradient_path is None or args.output is None or args.gradient_dim is None:
        raise ValueError("--gradient-path, --gradient-dim, and --output are required for --summary-only")
    device = torch.device(args.device)
    storage_dtype = np.float16 if args.gradient_dtype == "f16" else np.float32
    gradients = np.memmap(
        args.gradient_path,
        mode="r",
        dtype=storage_dtype,
        shape=(args.examples, args.gradient_dim),
    )
    scales = None
    if args.gradient_scale_path is not None:
        scales = np.memmap(args.gradient_scale_path, mode="r", dtype=np.float32, shape=(args.examples,))
    elif args.gradient_dtype == "f16":
        raise ValueError("--gradient-scale-path is required for f16 summaries")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats, top_directions, mean_direction = summarize_memmap(
        gradients,
        args.chunk_size,
        device,
        args.whole_gradient,
        scales,
        args.rank,
        args.summary_workspace or args.output.parent / f".{args.output.stem}_summary_workspace",
        args.directions_output,
        args.direction_output is not None or args.directions_output is not None,
        args.direction_dtype,
    )
    stats.update({
        "gradient_path": str(args.gradient_path),
        "gradient_storage": args.gradient_dtype,
        "summary_only": True,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    if args.direction_output:
        args.direction_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.direction_output, top_directions[0].astype(np.float32))
    if args.directions_output:
        top_directions.flush()
    if args.mean_direction_output:
        args.mean_direction_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.mean_direction_output, mean_direction.astype(np.float32))
    return stats


def self_check() -> None:
    gradients = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    stats, directions, mean = summarize(gradients)
    assert stats["fisher_lambda2"] < 1e-6
    assert stats["rank1_relative_frobenius_error"] < 1e-6
    assert stats["lambda_mean_over_trace"] <= 1.0 + 1e-6
    assert directions.shape == (1, 2)
    assert 0.0 <= stats["top_mean_direction_cosine_abs"] <= 1.0
    assert abs(stats["top_mean_direction_cosine_abs"] - 1.0) < 1e-6
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        gradient_path = path / "g.f16"
        scales_path = path / "s.f32"
        np.memmap(gradient_path, mode="w+", dtype=np.float16, shape=(3, 2))[:] = np.array(
            [[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]], dtype=np.float16
        )
        np.memmap(scales_path, mode="w+", dtype=np.float32, shape=(3,))[:] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        stats2, directions2, _ = summarize_memmap(
            np.memmap(gradient_path, mode="r", dtype=np.float16, shape=(3, 2)),
            chunk_size=2,
            rank=1,
            scales=np.memmap(scales_path, mode="r", dtype=np.float32, shape=(3,)),
            workspace=path / "ws",
            store_directions=False,
        )
        assert directions2 is None
        assert stats2["fisher_lambda2"] < 1e-6
    print("fisher diagnostic self-check: ok")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/mdm_safetensors/mdm-1028M-1600e18.safetensors")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--reverse-dir", type=Path, default=REVERSE_DIR)
    parser.add_argument("--a-cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--direction-output", type=Path)
    parser.add_argument("--directions-output", type=Path)
    parser.add_argument("--direction-dtype", choices=("f32", "f16"), default="f32")
    parser.add_argument("--mean-direction-output", type=Path)
    parser.add_argument("--scratch-dir", type=Path, default=ROOT / "runs/fisher_diagnostic")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--summary-workspace", type=Path)
    parser.add_argument("--gradient-path", type=Path)
    parser.add_argument("--gradient-scale-path", type=Path)
    parser.add_argument("--gradient-dim", type=int)
    parser.add_argument("--whole-gradient", action="store_true")
    parser.add_argument("--keep-gradients", action="store_true")
    parser.add_argument("--model", type=int, default=1028)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trainable", choices=("all", "last_mlp", "last_block"), default="all")
    parser.add_argument("--task", choices=("p2d", "d2p"), default="d2p")
    parser.add_argument("--group-start", type=int, default=0)
    parser.add_argument("--group-count", type=int, default=2)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=16_777_216)
    parser.add_argument("--gradient-dtype", choices=("f32", "f16"), default="f32")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--mask-min", type=float, default=1e-3)
    parser.add_argument("--mask-max", type=float, default=1.0)
    parser.add_argument("--normalization", choices=("mean", "sequence"), default="mean")
    parser.add_argument("--force-mask", action="store_true")
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fisher-seed", type=int, default=3428)
    parser.add_argument("--row-seed", type=int)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_check:
        self_check()
    elif arguments.summary_only:
        print(json.dumps(summarize_existing(arguments), indent=2, sort_keys=True), flush=True)
    else:
        print(json.dumps(estimate(arguments), indent=2, sort_keys=True), flush=True)
