"""A -> B -> A study on the official SMDM reverse-curse benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .continual_benchmark import (
        REVERSE_DIR,
        encode_benchmark_rows,
        read_jsonl,
        reverse_predictions,
        score_reverse,
        select,
    )
    from .continual_mdm import (
        RANKK_DIRECTION_BLOCK_SIZE,
        estimate_fisher,
        flat_parameters,
        load_model,
        load_cache,
        parameter_delta_stats,
        save_cache,
        set_seed,
        train_stage,
        trainable_parameters,
    )
except ImportError:
    from iclr_1.continual_benchmark import (
        REVERSE_DIR,
        encode_benchmark_rows,
        read_jsonl,
        reverse_predictions,
        score_reverse,
        select,
    )
    from iclr_1.continual_mdm import (
        RANKK_DIRECTION_BLOCK_SIZE,
        estimate_fisher,
        flat_parameters,
        load_model,
        load_cache,
        parameter_delta_stats,
        save_cache,
        set_seed,
        train_stage,
        trainable_parameters,
    )


ROOT = Path(__file__).resolve().parent


def direction_rows(directory: Path, direction: str, split: str, size: int, seed: int) -> list[dict]:
    rows = read_jsonl(directory / f"{direction}_prompts_{split}.jsonl")
    rows = [
        {"prompt": row["prompt"], "answer": row["completion"], "item_id": index}
        for index, row in enumerate(rows)
    ]
    return select(rows, size, seed)


def eval_rows(directory: Path, direction: str, size: int, seed: int) -> list[dict]:
    rows = read_jsonl(directory / f"{direction}_prompts_test.jsonl")
    return select(
        [
            {"prompt": row["prompt"], "target": row["completion"], "item_id": index}
            for index, row in enumerate(rows)
        ],
        size,
        seed,
    )


def fact_rows(directory: Path, direction: str, split: str, start: int, count: int) -> list[dict]:
    rows = read_jsonl(directory / f"{direction}_prompts_{split}.jsonl")
    group_size = 30 if split == "train" else 10
    left, right = start * group_size, (start + count) * group_size
    selected = rows[left:right]
    if len(selected) != count * group_size:
        raise ValueError(f"fact groups [{start}, {start + count}) exceed {direction} {split} data")
    result = []
    for offset, row in enumerate(selected):
        item = {
            "prompt": row["prompt"],
            "fact_id": start + offset // group_size,
            "template_id": offset % group_size,
            "item_id": left + offset,
        }
        item["answer" if split == "train" else "target"] = row["completion"]
        result.append(item)
    return result


def prediction_items(rows: list[dict], predictions: list[str]) -> list[dict]:
    items = []
    for row, prediction in zip(rows, predictions):
        target = row["target"]
        normalized_target = target.strip().lower()
        normalized_prediction = prediction.strip().lower()
        item = {
            "prompt": row["prompt"],
            "target": target,
            "prediction": prediction,
            "correct": normalized_target in normalized_prediction,
            "strict_correct": normalized_target == normalized_prediction,
        }
        for key in ("fact_id", "template_id", "item_id"):
            if key in row:
                item[key] = row[key]
        items.append(item)
    return items


def fact_summary(items: list[dict]) -> list[dict]:
    groups = {}
    for item in items:
        key = item.get("fact_id", item.get("item_id"))
        groups.setdefault(key, []).append(item)
    summary = []
    for key in sorted(groups, key=lambda value: (str(type(value)), value)):
        group = groups[key]
        correct = sum(item["correct"] for item in group)
        strict_correct = sum(item["strict_correct"] for item in group)
        summary.append(
            {
                "fact_id": key,
                "correct": correct,
                "strict_correct": strict_correct,
                "total": len(group),
                "accuracy": correct / len(group),
                "strict_accuracy": strict_correct / len(group),
            }
        )
    return summary


def measure(model, tokenizer, rows, args, device) -> tuple[dict, list[dict]]:
    predictions = reverse_predictions(model, tokenizer, rows, args, device)
    score = score_reverse(predictions, rows)
    if args.show_predictions:
        print(f"prediction={predictions[0]!r} target={rows[0]['target']!r}", flush=True)
    return score, prediction_items(rows, predictions)


def cache_metadata(args, fisher_min=None, fisher_max=None) -> dict:
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "reverse_dir": str(args.reverse_dir.resolve()),
        "model": args.model,
        "task_a": args.task_a,
        "task_b": args.task_b,
        "fact_split": args.fact_split,
        "a_group_start": args.a_group_start,
        "b_group_start": args.b_group_start,
        "group_count": args.group_count,
        "train_size": args.train_size,
        "data_seed": args.data_seed,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "a_steps": args.a_steps,
        "lr": args.lr,
        "seed": args.seed,
        "trainable": args.trainable,
    }
    if fisher_min is not None:
        metadata.update(
            {
                "fisher_batch_size": args.fisher_batch_size,
                "fisher_examples": args.fisher_examples,
                "fisher_seed": args.seed + 21,
                "fisher_mask_min": fisher_min,
                "fisher_mask_max": fisher_max,
                "fisher_answer_only": args.fisher_answer_only,
            }
        )
    return metadata


def check_cache(payload: dict, kind: str, expected: dict) -> None:
    if payload.get("kind") != kind:
        raise ValueError(f"cache kind mismatch: expected {kind!r}")
    actual = payload.get("metadata", {})
    mismatches = [
        f"{key}: cached={actual.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if mismatches:
        raise ValueError("cache metadata mismatch: " + "; ".join(mismatches))


def save_a_cache(model, path: Path, metadata: dict) -> None:
    save_cache(
        path,
        {"kind": "smdm_a_state_v1", "metadata": metadata, "state_dict": model.state_dict()},
    )


def save_fisher_cache(fisher: dict, stats: dict, path: Path, metadata: dict) -> None:
    tensors = {
        key: value.detach().cpu()
        for key, value in fisher.items()
        if isinstance(value, torch.Tensor)
    }
    values = {key: value for key, value in fisher.items() if not isinstance(value, torch.Tensor)}
    save_cache(
        path,
        {
            "kind": "smdm_fisher_v1",
            "metadata": metadata,
            "tensors": tensors,
            "values": values,
            "stats": stats,
        },
    )


def load_fisher_cache(payload: dict, device: torch.device, keep: tuple[str, ...]) -> tuple[dict, dict]:
    fisher = dict(payload["values"])
    tensors = payload["tensors"]
    for key in keep:
        if key in tensors:
            fisher[key] = tensors[key].to(device)
    if "u_top" not in fisher and "u" in tensors:
        fisher["u_top"] = tensors["u"].to(device)
    return fisher, payload["stats"]


def load_rankk_fisher(args, device: torch.device, dimension: int) -> tuple[dict, dict]:
    if args.rankk_directions is None or args.rankk_diagnostic is None:
        raise ValueError("rankk requires --rankk-directions and --rankk-diagnostic")
    directions = np.load(args.rankk_directions, mmap_mode="r")
    if directions.ndim != 2:
        raise ValueError("rank-k directions must have shape [rank, parameter_dim]")
    rank = directions.shape[0] if args.rankk_rank is None else args.rankk_rank
    if not 1 <= rank <= directions.shape[0] or directions.shape[1] != dimension:
        raise ValueError(
            f"rank-k directions shape {directions.shape} does not match rank={rank}, dim={dimension}"
        )
    diagnostic = json.loads(args.rankk_diagnostic.read_text())
    spectrum = np.asarray(diagnostic.get("fisher_spectrum", []), dtype=np.float32)
    if len(spectrum) < rank:
        raise ValueError("rank-k diagnostic has fewer eigenvalues than requested rank")
    direction_scales = None
    direction_block_size = RANKK_DIRECTION_BLOCK_SIZE
    direction_device = torch.device("cpu") if args.rankk_cpu_directions else device
    if args.rankk_direction_dtype == "f8":
        direction_dtype = getattr(torch, "float8_e4m3fn", None)
        if direction_dtype is None:
            raise RuntimeError("PyTorch float8_e4m3fn is required for --rankk-direction-dtype f8")
        block_count = (dimension + direction_block_size - 1) // direction_block_size
        direction_tensor = torch.empty((rank, dimension), dtype=direction_dtype, device=direction_device)
        direction_scales = torch.empty((rank, block_count), dtype=torch.float32, device=direction_device)
        for block_index, start in enumerate(range(0, dimension, direction_block_size)):
            stop = min(start + direction_block_size, dimension)
            block = np.asarray(directions[:rank, start:stop], dtype=np.float32)
            scales = np.maximum(np.abs(block).max(axis=1) / 448.0, 1e-12).astype(np.float32)
            encoded = torch.from_numpy(block / scales[:, None]).to(device=direction_device, dtype=direction_dtype)
            direction_tensor[:, start:stop] = encoded
            direction_scales[:, block_index] = torch.from_numpy(scales).to(direction_device)
    else:
        direction_tensor = torch.from_numpy(np.array(directions[:rank], copy=True)).to(direction_device)
    if args.rankk_cpu_directions:
        # ponytail: pin CPU directions so chunk-wise non_blocking copies can overlap transfer; if this still stalls, the next cut is a different storage layout.
        direction_tensor = direction_tensor.pin_memory()
        if direction_scales is not None:
            direction_scales = direction_scales.pin_memory()
    lambda_tensor = torch.from_numpy(spectrum[:rank].copy()).to(device=device, dtype=torch.float32)
    stats = dict(diagnostic)
    stats.update(
        {
            "fisher_rank": rank,
            "rankk_direction_dtype": args.rankk_direction_dtype,
            "rankk_cpu_directions": args.rankk_cpu_directions,
            "rankk_direction_block_size": direction_block_size,
            "rankk_directions": str(args.rankk_directions),
            "rankk_diagnostic": str(args.rankk_diagnostic),
        }
    )
    return {
        "u_top": direction_tensor[0],
        "u_top_k": direction_tensor,
        "u_top_k_scales": direction_scales,
        "u_top_k_block_size": direction_block_size,
        "lambda1": float(spectrum[0]),
        "lambda_k": lambda_tensor,
        "lambda_mean": float(diagnostic.get("fisher_lambda_mean", spectrum[0])),
    }, stats


def displacement_stats(parameters, theta_ref, fisher) -> dict:
    directions = {}
    if fisher is not None:
        directions = {
            "top": fisher.get("u_top", fisher.get("u")),
            "mean": fisher.get("u_mean"),
        }
    direction_scales = None
    if fisher is not None and fisher.get("u_top_k_scales") is not None:
        direction_scales = {"top": fisher["u_top_k_scales"][0]}
    return parameter_delta_stats(
        parameters,
        theta_ref,
        directions,
        direction_scales,
        fisher.get("u_top_k_block_size", RANKK_DIRECTION_BLOCK_SIZE)
        if fisher is not None
        else RANKK_DIRECTION_BLOCK_SIZE,
    )


def zero_fisher_stats() -> dict:
    return {
        "fisher_examples": 0,
        "fisher_dim": 0,
        "fisher_trace": 0.0,
        "fisher_lambda1": 0.0,
        "fisher_lambda2": 0.0,
        "fisher_lambda_mean": 0.0,
        "lambda1_over_trace": 0.0,
        "lambda2_over_lambda1": 0.0,
        "lambda_mean_over_trace": 0.0,
        "rank1_relative_frobenius_error": 0.0,
        "mean_rank1_relative_frobenius_error": 0.0,
        "diagonal_relative_frobenius_error": 0.0,
        "gradient_pairwise_cosine": 0.0,
        "gradient_pairwise_cosine_abs": 0.0,
        "gradient_mean_cosine": 0.0,
        "gradient_mean_cosine_abs": 0.0,
        "top_mean_direction_cosine_abs": 0.0,
        "fisher_spectrum": [],
    }


def merge_fisher_stats(result: dict, stats: dict) -> None:
    for key, value in stats.items():
        output_key = key
        if output_key in result:
            output_key = f"fisher_stat_{key}"
            while output_key in result:
                output_key = f"fisher_stat_{output_key}"
        result[output_key] = value


def run(args) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_float32_matmul_precision("high")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    if args.task_a == args.task_b:
        raise ValueError("task_a and task_b must be different")
    if args.fact_split:
        a_train_raw = fact_rows(args.reverse_dir, args.task_a, "train", args.a_group_start, args.group_count)
        b_train_raw = fact_rows(args.reverse_dir, args.task_b, "train", args.b_group_start, args.group_count)
        a_eval = fact_rows(args.reverse_dir, args.task_a, "test", args.a_group_start, args.group_count)
        b_eval = fact_rows(args.reverse_dir, args.task_b, "test", args.b_group_start, args.group_count)
    else:
        a_train_raw = direction_rows(args.reverse_dir, args.task_a, "train", args.train_size, args.data_seed)
        b_train_raw = direction_rows(args.reverse_dir, args.task_b, "train", args.train_size, args.data_seed)
        a_eval = eval_rows(args.reverse_dir, args.task_a, args.eval_size, args.eval_seed)
        b_eval = eval_rows(args.reverse_dir, args.task_b, args.eval_size, args.eval_seed)
    a_train = encode_benchmark_rows(a_train_raw, tokenizer, args.max_length)
    b_train = encode_benchmark_rows(b_train_raw, tokenizer, args.max_length)

    model = load_model(args, device)
    parameters = trainable_parameters(model, args.trainable)
    print(f"device={device} trainable={sum(p.numel() for p in parameters):,}", flush=True)
    a_metadata = cache_metadata(args)
    a_cache_metadata = dict(a_metadata)
    a_cache_metadata.pop("trainable", None)
    a_cache_metadata.pop("lr", None)
    if args.a_cache and args.a_cache.exists():
        payload = load_cache(args.a_cache, device)
        check_cache(payload, "smdm_a_state_v1", a_cache_metadata)
        model.load_state_dict(payload["state_dict"])
        del payload
        a_training_stats = {"loaded_from_cache": True, "steps": args.a_steps}
        print(f"loaded_a_cache={args.a_cache}", flush=True)
    else:
        a_training_stats = train_stage(
            model, a_train, parameters, device, args.batch_size, args.a_steps, args.lr,
            "plain", None, None, args.clip, args.seed + 1, pad_id, 0.0,
        )
        if args.a_cache:
            save_a_cache(model, args.a_cache, a_metadata)
            print(f"saved_a_cache={args.a_cache}", flush=True)
    a_after_a, a_after_a_items = measure(model, tokenizer, a_eval, args, device)
    b_before_b, b_before_b_items = measure(model, tokenizer, b_eval, args, device)
    theta_ref = flat_parameters(parameters).detach()
    if args.method == "rankk":
        # ponytail: rank-k reference uses fp16 to fit 1028M full-model state on 40GB; shard state if precision matters.
        theta_ref = theta_ref.to(dtype=torch.float16)
    else:
        theta_ref = theta_ref.clone()

    fisher_min = (
        0.8
        if args.method in ("rank1_low_snr", "mean_rank1_low_snr") and args.fisher_mask_min is None
        else args.fisher_mask_min
    )
    fisher_min = 1e-3 if fisher_min is None else fisher_min
    fisher_max = 1.0 if args.fisher_mask_max is None else args.fisher_mask_max
    if not 0.0 <= fisher_min < fisher_max <= 1.0:
        raise ValueError("fisher mask window must satisfy 0 <= min < max <= 1")
    if args.method == "plain":
        fisher = None
        fisher_stats = zero_fisher_stats()
    elif args.method == "rankk":
        fisher, fisher_stats = load_rankk_fisher(args, device, theta_ref.numel())
        print(f"loaded_rankk_directions={args.rankk_directions}", flush=True)
    else:
        fisher_metadata = cache_metadata(args, fisher_min, fisher_max)
        if args.fisher_cache and args.fisher_cache.exists():
            payload = load_cache(args.fisher_cache, torch.device("cpu"))
            check_cache(payload, "smdm_fisher_v1", fisher_metadata)
            keep = ("u_top", "u_mean")
            if args.method == "diagonal":
                keep = ("diag", "u_top", "u_mean")
            fisher, fisher_stats = load_fisher_cache(payload, device, keep)
            del payload
            print(f"loaded_fisher_cache={args.fisher_cache}", flush=True)
        else:
            fisher, fisher_stats = estimate_fisher(
                model, a_train, parameters, device, args.fisher_batch_size,
                args.fisher_examples, pad_id, args.seed + 21, fisher_min, fisher_max,
                args.fisher_answer_only,
            )
            if args.fisher_cache:
                save_fisher_cache(fisher, fisher_stats, args.fisher_cache, fisher_metadata)
                print(f"saved_fisher_cache={args.fisher_cache}", flush=True)
            if args.method in ("rank1", "rank1_low_snr", "mean_rank1", "mean_rank1_low_snr"):
                fisher.pop("u", None)
                fisher.pop("diag", None)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    ewc_lambda = args.ewc_lambda
    ewc_coefficient = None
    if fisher is not None:
        if args.method in ("rank1", "rank1_low_snr"):
            ewc_coefficient = fisher["lambda1"]
        elif args.method in ("mean_rank1", "mean_rank1_low_snr"):
            ewc_coefficient = fisher["lambda_mean"]
        elif args.method == "rankk":
            ewc_coefficient = float(fisher["lambda_k"].sum().cpu())
        if args.ewc_target_stiffness is not None:
            if ewc_coefficient is None:
                raise ValueError("--ewc-target-stiffness requires a low-rank Fisher method")
            ewc_lambda = args.ewc_target_stiffness / max(ewc_coefficient, 1e-12)
    elif args.ewc_target_stiffness is not None:
        raise ValueError("--ewc-target-stiffness requires an EWC method")
    b_training_stats = train_stage(
        model, b_train, parameters, device, args.batch_size, args.b_steps, args.lr,
        args.method, fisher, theta_ref, args.clip, args.seed + 2, pad_id, ewc_lambda,
        replay_rows=a_train if args.replay_weight > 0 else None,
        replay_weight=args.replay_weight,
    )
    displacement = displacement_stats(parameters, theta_ref, fisher)
    a_after_b, a_after_b_items = measure(model, tokenizer, a_eval, args, device)
    b_after_b, b_after_b_items = measure(model, tokenizer, b_eval, args, device)
    result = {
        "benchmark": "smdm_reverse_curse",
        "task_a": args.task_a,
        "task_b": args.task_b,
        "method": args.method,
        "seed": args.seed,
        "model": f"Diff_LLaMA_{args.model}M",
        "trainable": args.trainable,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "train_size_per_stage": len(a_train_raw),
        "eval_size_per_direction": len(a_eval),
        "fact_split": args.fact_split,
        "a_group_start": args.a_group_start,
        "b_group_start": args.b_group_start,
        "fact_group_count": args.group_count,
        "a_after_a": a_after_a,
        "a_after_a_items": a_after_a_items,
        "a_after_a_by_fact": fact_summary(a_after_a_items),
        "a_after_b": a_after_b,
        "a_after_b_items": a_after_b_items,
        "a_after_b_by_fact": fact_summary(a_after_b_items),
        "a_forgetting_absolute": a_after_a["accuracy"] - a_after_b["accuracy"],
        "b_before_b": b_before_b,
        "b_before_b_items": b_before_b_items,
        "b_before_b_by_fact": fact_summary(b_before_b_items),
        "b_after_b": b_after_b,
        "b_after_b_items": b_after_b_items,
        "b_after_b_by_fact": fact_summary(b_after_b_items),
        "b_gain_absolute": b_after_b["accuracy"] - b_before_b["accuracy"],
        "a_training": a_training_stats,
        "b_training": b_training_stats,
        "replay_weight": args.replay_weight,
        "a_cache": str(args.a_cache) if args.a_cache else None,
        "fisher_cache": str(args.fisher_cache) if args.fisher_cache else None,
        "fisher_answer_only": args.fisher_answer_only,
        "a_steps": args.a_steps,
        "b_steps": args.b_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "reverse_steps": args.reverse_steps,
        "generation_batch_size": args.generation_batch_size,
        "generation_seed": args.generation_seed,
        "ewc_lambda": 0.0 if args.method == "plain" else ewc_lambda,
        "ewc_lambda_requested": 0.0 if args.method == "plain" else args.ewc_lambda,
        "ewc_target_stiffness": args.ewc_target_stiffness,
        "ewc_effective_stiffness": 0.0 if ewc_coefficient is None else ewc_lambda * ewc_coefficient,
        "fisher_mask_min": fisher_min,
        "fisher_mask_max": fisher_max,
    }
    result.update(displacement)
    merge_fisher_stats(result, fisher_stats)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--reverse-dir", type=Path, default=REVERSE_DIR)
    parser.add_argument("--task-a", choices=("p2d", "d2p"), default="p2d")
    parser.add_argument("--task-b", choices=("p2d", "d2p"), default="d2p")
    parser.add_argument("--fact-split", action="store_true")
    parser.add_argument("--a-group-start", type=int, default=0)
    parser.add_argument("--b-group-start", type=int, default=5)
    parser.add_argument("--group-count", type=int, default=5)
    parser.add_argument("--model", type=int, default=170)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--method",
        choices=("plain", "rank1", "rank1_low_snr", "mean_rank1", "mean_rank1_low_snr", "diagonal", "rankk"),
        default="plain",
    )
    parser.add_argument("--trainable", choices=("all", "last_mlp", "last_block"), default="all")
    parser.add_argument("--train-size", type=int, default=450)
    parser.add_argument("--eval-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--fisher-batch-size", type=int, default=2)
    parser.add_argument("--fisher-examples", type=int, default=4)
    parser.add_argument("--a-steps", type=int, default=500)
    parser.add_argument("--b-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ewc-lambda", type=float, default=30.0)
    parser.add_argument("--replay-weight", type=float, default=0.0)
    parser.add_argument("--ewc-target-stiffness", type=float)
    parser.add_argument("--fisher-mask-min", type=float)
    parser.add_argument("--fisher-mask-max", type=float)
    parser.add_argument("--fisher-answer-only", action="store_true")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--data-seed", type=int, default=3407)
    parser.add_argument("--eval-seed", type=int, default=3407)
    parser.add_argument("--reverse-steps", type=int, default=32)
    parser.add_argument("--reverse-length", type=int, default=52)
    parser.add_argument("--reverse-cfg", type=float, default=0.8)
    parser.add_argument("--reverse-temperature", type=float, default=0.0)
    parser.add_argument("--generation-seed", type=int, default=3407)
    parser.add_argument("--show-predictions", action="store_true")
    parser.add_argument("--a-cache", type=Path)
    parser.add_argument("--fisher-cache", type=Path)
    parser.add_argument("--rankk-directions", type=Path)
    parser.add_argument("--rankk-diagnostic", type=Path)
    parser.add_argument("--rankk-rank", type=int)
    parser.add_argument("--rankk-direction-dtype", choices=("f16", "f8"), default="f16")
    parser.add_argument("--rankk-cpu-directions", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
