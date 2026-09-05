#!/usr/bin/env python3
"""Fail-closed contract for the paired all-target/one-target comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONFIG_FIELDS = (
    "model_size",
    "mask_probabilities",
    "sample_sizes",
    "test_samples",
    "sequence_length",
    "parameter",
    "shuffle_records",
    "include_native_schedule",
    "native_eps",
    "split",
)


def _read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def _load(paths):
    payloads = {}
    for path in paths:
        payload = _read_json(path)
        if payload.get("status") != "ok" or payload.get("schema_version") != 2:
            raise ValueError(f"comparison evidence is not an audited success: {path}")
        if "comparison_audit" not in payload:
            raise ValueError(f"comparison audit metadata missing: {path}")
        seed = payload["config"]["seed"]
        if seed in payloads:
            raise ValueError(f"duplicate seed {seed}: {path}")
        payloads[seed] = (path, payload)
    if sorted(payloads) != [0, 1, 2]:
        raise ValueError("comparison groups require seeds 0, 1, and 2")
    return payloads


def _grid(payload):
    return {
        (row["sample_count"], row["test_sample_count"], row["parameter"], row["mask_condition"])
        for row in payload["results"]
    }


def _critical(payload):
    return {
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "source_sha256": payload["source_sha256"],
        "probe_sha256": payload["probe_sha256"],
        "base_probe_sha256": payload.get("base_probe_sha256"),
        **{field: payload["config"].get(field) for field in CONFIG_FIELDS},
    }


def _primary_errors(stats):
    calibration_gram = stats["calibration_gram"]
    test_gram = stats["test_gram"]
    cross = stats["test_by_calibration_inner_products"]
    calibration_diagonal = stats["calibration_diagonal"]
    test_diagonal = stats["test_diagonal"]
    calibration_count = len(calibration_gram)
    test_count = len(test_gram)
    test_norm_sq = sum(value * value for row in test_gram for value in row)
    gram_ones = [sum(row) for row in calibration_gram]
    mean_norm_sq = sum(gram_ones) / calibration_count
    mean_fisher_quadratic = sum(value * value for value in gram_ones) / calibration_count
    coefficient = mean_fisher_quadratic / max(mean_norm_sq * mean_norm_sq, 1e-30)
    test_projection = [sum(row) / calibration_count for row in cross]
    mean_test_quadratic = sum(value * value for value in test_projection) / test_count
    rank1_residual = test_norm_sq - 2 * coefficient * mean_test_quadratic + coefficient * coefficient * mean_norm_sq * mean_norm_sq
    diagonal_inner = sum(left * right for left, right in zip(calibration_diagonal, test_diagonal))
    diagonal_norm_sq = sum(value * value for value in calibration_diagonal)
    diagonal_residual = test_norm_sq - 2 * diagonal_inner + diagonal_norm_sq
    denominator = math.sqrt(max(test_norm_sq, 1e-30))
    return (
        math.sqrt(max(rank1_residual, 0.0)) / denominator,
        math.sqrt(max(diagonal_residual, 0.0)) / denominator,
    )


def _verify_sufficient_statistics(payload, path):
    for row in payload["results"]:
        stats = row.get("audit_sufficient_statistics")
        if not stats:
            raise ValueError(f"sufficient statistics missing: {path}")
        rank1, diagonal = _primary_errors(stats)
        if not math.isclose(rank1, row["mean_rank1_test_relative_frobenius_error"], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"rank-1 metric is not reproducible from sufficient statistics: {path}")
        if not math.isclose(diagonal, row["diagonal_test_relative_frobenius_error"], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"diagonal metric is not reproducible from sufficient statistics: {path}")


def build(all_paths, one_paths):
    groups = {
        "gsm8k_all_targets": _load(all_paths),
        "gsm8k_one_target": _load(one_paths),
    }
    expected_modes = {
        "gsm8k_all_targets": (None, "native_conditional"),
        "gsm8k_one_target": (None, "fixed_target"),
    }
    for name, members in groups.items():
        expected_task, expected_loss = expected_modes[name]
        for path, payload in members.values():
            if payload["config"].get("task_id") != expected_task or payload["config"].get("loss_mode") != expected_loss:
                raise ValueError(f"unexpected task/loss contract: {path}")
    for members in groups.values():
        for path, payload in members.values():
            _verify_sufficient_statistics(payload, path)

    objective_pairs = []
    for seed in (0, 1, 2):
        all_path, all_payload = groups["gsm8k_all_targets"][seed]
        one_path, one_payload = groups["gsm8k_one_target"][seed]
        if _critical(all_payload) != _critical(one_payload):
            raise ValueError(f"objective critical configuration mismatch at seed {seed}")
        if all_payload["data_sha256"] != one_payload["data_sha256"] or _grid(all_payload) != _grid(one_payload):
            raise ValueError(f"objective data/grid mismatch at seed {seed}")
        if all_payload["config"]["loss_mode"] != "native_conditional" or one_payload["config"]["loss_mode"] != "fixed_target":
            raise ValueError(f"objective modes are not all-target versus one-target at seed {seed}")
        if all_payload["comparison_audit"] != one_payload["comparison_audit"]:
            raise ValueError(f"objective rows or masks are not exactly paired at seed {seed}")
        objective_pairs.append({
            "seed": seed,
            "all_target": _relative(all_path),
            "one_target": _relative(one_path),
            "calibration_records_sha256": all_payload["comparison_audit"]["selected_records"]["calibration_sha256"],
            "test_records_sha256": all_payload["comparison_audit"]["selected_records"]["test_sha256"],
            "mask_hashes": all_payload["comparison_audit"]["mask_hashes"],
        })

    artifacts = []
    for name, members in groups.items():
        for seed, (path, payload) in members.items():
            artifacts.append({
                "group": name,
                "seed": seed,
                "path": _relative(path),
                "sha256": _sha256(path),
                "probe_sha256": payload["probe_sha256"],
            })
    return {
        "schema_version": 1,
        "status": "ok",
        "contract_script_sha256": _sha256(__file__),
        "artifacts": sorted(artifacts, key=lambda item: (item["group"], item["seed"])),
        "objective_pairs": objective_pairs,
        "verified": [
            "identical calibration/test sizes and grids",
            "identical objective-pair record identities and corrupted masks",
            "expected task IDs and loss modes for both comparison groups",
            "identical audited-wrapper and frozen base-probe hashes",
            "rank-1 and diagonal errors recomputed from stored sufficient statistics",
        ],
    }


def validate_contract(path):
    payload = _read_json(path)
    by_group = {}
    for artifact in payload["artifacts"]:
        by_group.setdefault(artifact["group"], []).append(ROOT / artifact["path"])
        if _sha256(ROOT / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"comparison artifact hash mismatch: {artifact['path']}")
    rebuilt = build(by_group["gsm8k_all_targets"], by_group["gsm8k_one_target"])
    if rebuilt != payload:
        raise ValueError("comparison contract is stale")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-target", type=Path, nargs="+")
    parser.add_argument("--one-target", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        stats = {
            "calibration_gram": [[1.0, 0.0], [0.0, 1.0]],
            "test_gram": [[1.0, 0.0], [0.0, 1.0]],
            "test_by_calibration_inner_products": [[2.0, 0.0], [0.0, 2.0]],
            "calibration_diagonal": [1.0, 1.0],
            "test_diagonal": [1.0, 1.0],
        }
        rank1, diagonal = _primary_errors(stats)
        assert math.isfinite(rank1) and diagonal == 0.0
        print(json.dumps({"self_check": "ok"}))
        return
    if not all((args.all_target, args.one_target, args.output)):
        parser.error("both objective groups and --output are required")
    result = build(args.all_target, args.one_target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "artifacts": len(result["artifacts"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
