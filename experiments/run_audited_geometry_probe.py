#!/usr/bin/env python3
"""Run the frozen geometry probe while retaining comparison audit data."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path


ROOT = Path(__file__).parents[1]
BASE_PROBE = Path(__file__).with_name("dllm_rank1_probe.py")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _selected_records(args, maximum, test_count):
    records = []
    with Path(args.data).open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if args.split and record.get("split") != args.split:
                continue
            if args.task_id is not None and int(record.get("task_id", -1)) != args.task_id:
                continue
            input_ids = [int(token) for token in record["input_ids"]]
            records.append({
                "source_line": source_line,
                "input_ids_sha256": _json_hash(input_ids),
            })
    if args.shuffle_records:
        random.Random(args.seed).shuffle(records)
    selected = records[: maximum + test_count]
    if len(selected) != maximum + test_count:
        raise ValueError("audit record selection is incomplete")
    return {
        "calibration": selected[:maximum],
        "test": selected[maximum:],
        "calibration_sha256": _json_hash(selected[:maximum]),
        "test_sha256": _json_hash(selected[maximum:]),
    }


def main():
    spec = importlib.util.spec_from_file_location("frozen_dllm_rank1_probe", BASE_PROBE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    base_split_metrics = probe._split_metrics
    base_sample_masks = probe._sample_masks
    base_run = probe._run
    captured_masks = []

    def audited_split_metrics(calibration_gradients, test_gradients, calibration_losses, test_losses):
        import torch

        result = base_split_metrics(calibration_gradients, test_gradients, calibration_losses, test_losses)
        calibration = torch.stack(calibration_gradients).to(dtype=torch.float64)
        test = torch.stack(test_gradients).to(dtype=torch.float64)
        # ponytail: Gram, cross-Gram, and diagonals are the smallest exact
        # sufficient bundle for recomputing the paper's rank-1/diagonal errors.
        result["audit_sufficient_statistics"] = {
            "calibration_gram": ((calibration @ calibration.T) / len(calibration)).tolist(),
            "test_gram": ((test @ test.T) / len(test)).tolist(),
            "test_by_calibration_inner_products": (test @ calibration.T).tolist(),
            "calibration_diagonal": torch.mean(calibration * calibration, dim=0).tolist(),
            "test_diagonal": torch.mean(test * test, dim=0).tolist(),
        }
        return result

    def audited_sample_masks(probabilities, length, device, generator):
        masks = base_sample_masks(probabilities, length, device, generator)
        captured_masks.append((probabilities.detach().cpu(), masks.detach().cpu()))
        return masks

    def audited_run(args):
        import torch

        result = base_run(args)
        sample_sizes = probe._parse_ints(args.sample_sizes) if args.sample_sizes else [args.samples]
        maximum = max(sample_sizes)
        required = maximum + args.test_samples
        conditions = [f"fixed_{value:g}" for value in probe._parse_floats(args.mask_probabilities)]
        if args.include_native_schedule:
            conditions.append("native_schedule")
        if len(captured_masks) != len(conditions):
            raise ValueError("audit mask capture does not match configured conditions")

        replay_generator = torch.Generator(device=args.device).manual_seed(args.seed)
        target_uniform = torch.rand((required, result["results"][0]["sequence_length"]), device=args.device, generator=replay_generator).cpu()
        masks = []
        for condition, (probabilities, values) in zip(conditions, captured_masks):
            selected_targets = torch.argmin(target_uniform.masked_fill(~values, float("inf")), dim=1)
            masks.append({
                "mask_condition": condition,
                "probabilities_sha256": _tensor_hash(probabilities),
                "calibration_mask_sha256": _tensor_hash(values[:maximum]),
                "test_mask_sha256": _tensor_hash(values[maximum:]),
                "selected_target_indices_sha256": _tensor_hash(selected_targets),
            })

        result["schema_version"] = 2
        result["base_probe_sha256"] = _sha256(BASE_PROBE)
        result["probe_sha256"] = _sha256(__file__)
        result["comparison_audit"] = {
            "selected_records": _selected_records(args, maximum, args.test_samples),
            "mask_hashes": masks,
            "sufficient_statistics": "stored in every result row",
        }
        return result

    probe._split_metrics = audited_split_metrics
    probe._sample_masks = audited_sample_masks
    probe._run = audited_run
    return probe.main()


if __name__ == "__main__":
    raise SystemExit(main())
