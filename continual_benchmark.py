"""A -> B -> A benchmark study for the SMDM masked diffusion LM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

try:
    from .continual_mdm import (
        MASK_ID,
        estimate_fisher,
        flat_parameters,
        load_model,
        set_seed,
        train_stage,
        trainable_parameters,
    )
except ImportError:
    from iclr_1.continual_mdm import (
        MASK_ID,
        estimate_fisher,
        flat_parameters,
        load_model,
        set_seed,
        train_stage,
        trainable_parameters,
    )


ROOT = Path(__file__).resolve().parent
GSM_TEST = ROOT / "SMDM/data/gsm8k/test.jsonl"
GSM_TRAIN = ROOT / "SMDM/data/gsm8k/train_no_aug.txt"
REVERSE_DIR = ROOT / "SMDM/data/reverse_experiments/june_version_7921032488"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(rows: list[dict], size: int, seed: int) -> list[dict]:
    if size <= 0:
        raise ValueError("size must be positive")
    if size >= len(rows):
        return rows[:]
    indices = list(range(len(rows)))
    generator = __import__("random").Random(seed)
    generator.shuffle(indices)
    return [rows[index] for index in indices[:size]]


def load_gsm_train(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if "||" not in line or "####" not in line:
                continue
            question, answer = line.rstrip("\n").split("||", 1)
            rationale, target = answer.rsplit("####", 1)
            rows.append(
                {
                    "prompt": "Question: " + question,
                    "answer": "Answer: " + rationale.strip() + "####" + target.strip(),
                }
            )
    if not rows:
        raise ValueError(f"no GSM8K rows found in {path}")
    return rows


def load_gsm_eval(path: Path, size: int, seed: int) -> list[dict]:
    rows = read_jsonl(path)
    return select(
        [{"prompt": "Question: " + row["question"], "target": row["target"]} for row in rows],
        size,
        seed,
    )


def load_reverse_train(directory: Path, size: int, seed: int) -> list[dict]:
    rows = []
    for name in ("p2d_prompts_train.jsonl", "d2p_prompts_train.jsonl"):
        rows.extend(
            {"prompt": row["prompt"], "answer": row["completion"]}
            for row in read_jsonl(directory / name)
        )
    return select(rows, size, seed)


def load_reverse_eval(directory: Path, size: int, seed: int) -> dict[str, list[dict]]:
    result = {}
    for direction in ("p2d", "d2p"):
        rows = read_jsonl(directory / f"{direction}_prompts_test.jsonl")
        result[direction] = select(
            [{"prompt": row["prompt"], "target": row["completion"]} for row in rows],
            size,
            seed,
        )
    return result


def encode_benchmark_rows(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    encoded = []
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
        ids = tokenizer(row["prompt"] + row["answer"], add_special_tokens=True)["input_ids"]
        ids = ids + [tokenizer.eos_token_id]
        if len(ids) <= max_length and len(ids) > len(prompt_ids):
            encoded.append(
                {
                    "ids": ids,
                    "answer_start": len(prompt_ids),
                    "answer_end": len(ids) - 1,
                }
            )
    if not encoded:
        raise ValueError("all benchmark training examples were removed by max_length")
    return encoded


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64).clamp_min(1e-12)
    return logits.exp() / (-torch.log(noise)).pow(temperature)


@torch.no_grad()
def diff_generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    steps: int,
    context_length: int,
    cfg_scale: float,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    if not prompts:
        raise ValueError("generation batch cannot be empty")
    prompt_ids = [tokenizer(prompt, add_special_tokens=True)["input_ids"] for prompt in prompts]
    width = max(len(ids) for ids in prompt_ids)
    if width > context_length:
        raise ValueError(f"prompt length {width} exceeds context length {context_length}")
    x = torch.full((len(prompts), context_length), MASK_ID, dtype=torch.long, device=device)
    for index, ids in enumerate(prompt_ids):
        x[index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    if width == context_length:
        return x, width

    timesteps = torch.linspace(1, 1e-5, steps + 1, device=device)
    model.eval()
    for step in range(steps):
        mask_index = x == MASK_ID
        if cfg_scale > 0:
            unconditional = x.clone()
            unconditional[:, :width] = MASK_ID
            logits = model(torch.cat([x, unconditional], dim=0))
            logits, unconditional_logits = torch.chunk(logits, 2, dim=0)
            logits = logits[mask_index]
            unconditional_logits = unconditional_logits[mask_index]
            logits = unconditional_logits + (cfg_scale + 1) * (logits - unconditional_logits)
        else:
            logits = model(x)[mask_index]

        t, s = timesteps[step], timesteps[step + 1]
        x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
        if step < steps - 1:
            number_transfer = int(mask_index.sum() * (1 - s / t))
            if number_transfer:
                probabilities = torch.softmax(logits.to(torch.float64), dim=-1)
                confidence = probabilities.gather(1, x0[:, None]).squeeze(1)
                _, indices = torch.topk(confidence, number_transfer)
                transferred = torch.full_like(x0, MASK_ID)
                transferred[indices] = x0[indices]
                x[mask_index] = transferred
        else:
            x[mask_index] = x0
    return x, width


def decode_rows(tokenizer, ids: torch.Tensor) -> list[str]:
    return [tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in ids]


def gsm_predictions(model, tokenizer, rows, args, device) -> list[str]:
    predictions = []
    for start in range(0, len(rows), args.generation_batch_size):
        batch = rows[start : start + args.generation_batch_size]
        set_seed(args.generation_seed + start)
        first, _ = diff_generate_batch(
            model,
            tokenizer,
            [row["prompt"] for row in batch],
            device,
            args.gsm_steps,
            args.gsm_length,
            args.gsm_cfg1,
            args.gsm_temperature,
        )
        prefixes = decode_rows(tokenizer, first)
        second, _ = diff_generate_batch(
            model,
            tokenizer,
            prefixes,
            device,
            args.gsm_steps,
            args.gsm_length,
            args.gsm_cfg2,
            args.gsm_temperature,
        )
        predictions.extend(decode_rows(tokenizer, second))
    return predictions


def reverse_predictions(model, tokenizer, rows, args, device) -> list[str]:
    predictions = []
    for start in range(0, len(rows), args.generation_batch_size):
        batch = rows[start : start + args.generation_batch_size]
        set_seed(args.generation_seed + 10000 + start)
        output, width = diff_generate_batch(
            model,
            tokenizer,
            [row["prompt"] for row in batch],
            device,
            args.reverse_steps,
            args.reverse_length,
            args.reverse_cfg,
            args.reverse_temperature,
        )
        predictions.extend(
            tokenizer.decode(output[index, width:].tolist(), skip_special_tokens=True)
            for index in range(len(batch))
        )
    return predictions


def gsm_correct(prediction: str, target: str) -> bool:
    from eval.math_normalization import check_sympy_equivalence, normalize_final_answer

    matches = re.findall(r"####\s*([^\n]+)", prediction)
    answer = matches[-1] if matches else ""
    return check_sympy_equivalence(normalize_final_answer(answer), normalize_final_answer(target))


def score_gsm(predictions: list[str], rows: list[dict]) -> dict:
    correct = sum(gsm_correct(prediction, row["target"]) for prediction, row in zip(predictions, rows))
    return {"accuracy": correct / len(rows), "correct": correct, "total": len(rows)}


def score_reverse(predictions: list[str], rows: list[dict]) -> dict:
    contains_correct = [
        row["target"].strip().lower() in prediction.strip().lower()
        for prediction, row in zip(predictions, rows)
    ]
    exact_correct = [
        row["target"].strip().lower() == prediction.strip().lower()
        for prediction, row in zip(predictions, rows)
    ]
    correct = sum(contains_correct)
    strict_correct = sum(exact_correct)
    return {
        "accuracy": correct / len(rows),
        "correct": correct,
        "strict_accuracy": strict_correct / len(rows),
        "strict_correct": strict_correct,
        "total": len(rows),
    }


def evaluate_benchmarks(model, tokenizer, gsm_rows, reverse_rows, args, device) -> dict:
    gsm_outputs = gsm_predictions(model, tokenizer, gsm_rows, args, device)
    gsm = score_gsm(gsm_outputs, gsm_rows)
    reverse = {}
    for direction, rows in reverse_rows.items():
        outputs = reverse_predictions(model, tokenizer, rows, args, device)
        reverse[direction] = score_reverse(outputs, rows)
        if args.show_predictions:
            print(f"{direction}_prediction={outputs[0]!r} target={rows[0]['target']!r}", flush=True)
    if args.show_predictions:
        print(f"gsm8k_prediction={gsm_outputs[0]!r} target={gsm_rows[0]['target']!r}", flush=True)
    reverse["average"] = {
        "accuracy": sum(reverse[key]["accuracy"] for key in ("p2d", "d2p")) / 2,
        "correct": reverse["p2d"]["correct"] + reverse["d2p"]["correct"],
        "total": reverse["p2d"]["total"] + reverse["d2p"]["total"],
    }
    return {"gsm8k": gsm, "reverse": reverse}


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


def run(args) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_float32_matmul_precision("high")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    a_train_raw = select(load_gsm_train(args.gsm_train), args.train_size, args.data_seed)
    b_train_raw = load_reverse_train(args.reverse_dir, args.train_size, args.data_seed)
    a_eval_raw = load_gsm_eval(args.gsm_test, args.a_eval_size, args.eval_seed)
    b_eval_raw = load_reverse_eval(args.reverse_dir, args.b_eval_size, args.eval_seed)
    a_train = encode_benchmark_rows(a_train_raw, tokenizer, args.max_length)
    b_train = encode_benchmark_rows(b_train_raw, tokenizer, args.max_length)

    model = load_model(args, device)
    parameters = trainable_parameters(model, args.trainable)
    print(f"device={device} trainable={sum(p.numel() for p in parameters):,}", flush=True)
    train_stage(
        model, a_train, parameters, device, args.batch_size, args.a_steps, args.lr,
        "plain", None, None, args.clip, args.seed + 1, pad_id, 0.0,
    )
    a_after_a = evaluate_benchmarks(model, tokenizer, a_eval_raw, b_eval_raw, args, device)
    theta_ref = flat_parameters(parameters).detach().clone()

    fisher_min = args.fisher_mask_min
    if args.method in ("rank1_low_snr", "mean_rank1_low_snr") and fisher_min is None:
        fisher_min = 0.8
    fisher_min = 1e-3 if fisher_min is None else fisher_min
    fisher_max = 1.0 if args.fisher_mask_max is None else args.fisher_mask_max
    if not 0.0 <= fisher_min < fisher_max <= 1.0:
        raise ValueError("fisher mask window must satisfy 0 <= min < max <= 1")
    if args.method == "plain":
        fisher, fisher_stats = {"u": None, "diag": None, "lambda1": 0.0}, zero_fisher_stats()
    else:
        fisher, fisher_stats = estimate_fisher(
            model, a_train, parameters, device, args.fisher_batch_size,
            args.fisher_examples, pad_id, args.seed + 21, fisher_min, fisher_max,
            args.fisher_answer_only,
        )
    train_stage(
        model, b_train, parameters, device, args.batch_size, args.b_steps, args.lr,
        args.method, fisher, theta_ref, args.clip, args.seed + 2, pad_id, args.ewc_lambda,
    )
    after_b = evaluate_benchmarks(model, tokenizer, a_eval_raw, b_eval_raw, args, device)
    result = {
        "benchmark": "gsm8k_and_reverse_curse",
        "method": args.method,
        "seed": args.seed,
        "model": f"Diff_LLaMA_{args.model}M",
        "trainable": args.trainable,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "train_size_per_stage": args.train_size,
        "gsm8k_eval_size": args.a_eval_size,
        "reverse_eval_size_per_direction": args.b_eval_size,
        "a_gsm8k_after_a": a_after_a["gsm8k"],
        "a_gsm8k_after_b": after_b["gsm8k"],
        "a_gsm8k_forgetting_absolute": a_after_a["gsm8k"]["accuracy"] - after_b["gsm8k"]["accuracy"],
        "b_reverse_before_b": a_after_a["reverse"],
        "b_reverse_after_b": after_b["reverse"],
        "ewc_lambda": 0.0 if args.method == "plain" else args.ewc_lambda,
        "fisher_mask_min": fisher_min,
        "fisher_mask_max": fisher_max,
        "gsm_steps": args.gsm_steps,
        "reverse_steps": args.reverse_steps,
        **fisher_stats,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def self_check() -> None:
    assert gsm_correct("Question... #### 18", "18")
    assert not gsm_correct("Question... #### 19", "18")
    assert score_reverse([" the named person is Ada"], [{"target": "Ada"}])["accuracy"] == 1.0
    print("benchmark self-check: ok")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--model", type=int, default=170)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--method",
        choices=("plain", "rank1", "rank1_low_snr", "mean_rank1", "mean_rank1_low_snr", "diagonal"),
        default="plain",
    )
    parser.add_argument("--trainable", choices=("all", "last_mlp", "last_block"), default="last_mlp")
    parser.add_argument("--gsm-train", type=Path, default=GSM_TRAIN)
    parser.add_argument("--gsm-test", type=Path, default=GSM_TEST)
    parser.add_argument("--reverse-dir", type=Path, default=REVERSE_DIR)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--a-eval-size", type=int, default=32)
    parser.add_argument("--b-eval-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--fisher-batch-size", type=int, default=2)
    parser.add_argument("--fisher-examples", type=int, default=16)
    parser.add_argument("--a-steps", type=int, default=300)
    parser.add_argument("--b-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ewc-lambda", type=float, default=30.0)
    parser.add_argument("--fisher-mask-min", type=float)
    parser.add_argument("--fisher-mask-max", type=float)
    parser.add_argument("--fisher-answer-only", action="store_true")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--data-seed", type=int, default=3407)
    parser.add_argument("--eval-seed", type=int, default=3407)
    parser.add_argument("--generation-seed", type=int, default=3407)
    parser.add_argument("--gsm-steps", type=int, default=256)
    parser.add_argument("--gsm-length", type=int, default=256)
    parser.add_argument("--gsm-cfg1", type=float, default=0.1)
    parser.add_argument("--gsm-cfg2", type=float, default=0.1)
    parser.add_argument("--gsm-temperature", type=float, default=0.1)
    parser.add_argument("--reverse-steps", type=int, default=32)
    parser.add_argument("--reverse-length", type=int, default=52)
    parser.add_argument("--reverse-cfg", type=float, default=0.8)
    parser.add_argument("--reverse-temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show-predictions", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        run(args)
