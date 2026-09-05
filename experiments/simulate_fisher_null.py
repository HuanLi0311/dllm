#!/usr/bin/env python3
"""Measure finite-sample Fisher-surrogate bias under isotropic gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def _one(calibration, test):
    import torch

    count, dimension = calibration.shape
    test_count = test.shape[0]
    calibration_gram = calibration @ calibration.T / count
    test_gram = test @ test.T / test_count
    calibration_norm_sq = torch.sum(calibration_gram * calibration_gram)
    test_norm_sq = torch.sum(test_gram * test_gram)
    calibration_mean = calibration.mean(dim=0)
    mean_norm_sq = torch.dot(calibration_mean, calibration_mean)
    mean_quadratic = torch.mean((calibration @ calibration_mean) ** 2)
    coefficient = mean_quadratic / torch.clamp(mean_norm_sq * mean_norm_sq, min=1e-30)
    rank1_norm_sq = coefficient * coefficient * mean_norm_sq * mean_norm_sq
    calibration_rank1 = torch.sqrt(torch.clamp(calibration_norm_sq - 2 * coefficient * mean_quadratic + rank1_norm_sq, min=0)) / torch.sqrt(calibration_norm_sq)
    test_rank1_quadratic = torch.mean((test @ calibration_mean) ** 2)
    split_rank1 = torch.sqrt(torch.clamp(test_norm_sq - 2 * coefficient * test_rank1_quadratic + rank1_norm_sq, min=0)) / torch.sqrt(test_norm_sq)
    population_rank1 = torch.sqrt(torch.clamp(dimension - 2 * coefficient * mean_norm_sq + rank1_norm_sq, min=0)) / math.sqrt(dimension)

    calibration_diagonal = torch.mean(calibration * calibration, dim=0)
    test_diagonal = torch.mean(test * test, dim=0)
    calibration_diagonal_error = torch.sqrt(torch.clamp(calibration_norm_sq - torch.dot(calibration_diagonal, calibration_diagonal), min=0)) / torch.sqrt(calibration_norm_sq)
    split_diagonal = torch.sqrt(torch.clamp(test_norm_sq - 2 * torch.dot(calibration_diagonal, test_diagonal) + torch.dot(calibration_diagonal, calibration_diagonal), min=0)) / torch.sqrt(test_norm_sq)
    population_diagonal = torch.linalg.vector_norm(calibration_diagonal - 1.0) / math.sqrt(dimension)

    eigenvalues, eigenvectors = torch.linalg.eigh(calibration_gram)
    lambda1 = torch.clamp(eigenvalues[-1], min=1e-30)
    top_direction = calibration.T @ eigenvectors[:, -1] / torch.sqrt(count * lambda1)
    split_top1 = torch.sqrt(torch.clamp(test_norm_sq - 2 * lambda1 * torch.mean((test @ top_direction) ** 2) + lambda1 * lambda1, min=0)) / torch.sqrt(test_norm_sq)
    population_top1 = torch.sqrt(torch.clamp(dimension - 2 * lambda1 + lambda1 * lambda1, min=0)) / math.sqrt(dimension)
    return {
        "in_sample_rank1": float(calibration_rank1),
        "in_sample_diagonal": float(calibration_diagonal_error),
        "split_rank1": float(split_rank1),
        "split_diagonal": float(split_diagonal),
        "population_rank1": float(population_rank1),
        "population_diagonal": float(population_diagonal),
        "split_calibration_top1": float(split_top1),
        "population_calibration_top1": float(population_top1),
    }


def _summary(rows):
    output = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        ordered = sorted(values)
        output[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "q05": ordered[int(0.05 * (len(ordered) - 1))],
            "q95": ordered[int(0.95 * (len(ordered) - 1))],
        }
    for prefix in ("in_sample", "split", "population"):
        output[f"{prefix}_rank1_win_rate"] = statistics.fmean(row[f"{prefix}_rank1"] < row[f"{prefix}_diagonal"] for row in rows)
    return output


def _analytic_scaled_identity(dimension):
    """Spectrum for g(x)=vec(xx^T) with x standard Gaussian."""
    return {
        "dimension": dimension,
        "top_eigenvalue_units": dimension + 2,
        "second_eigenvalue_units": 2,
        "second_over_first": 2 / (dimension + 2),
        "nonzero_rank": dimension * (dimension + 1) // 2,
        "best_rank1_relative_frobenius_error": math.sqrt(2 * (dimension - 1) / (3 * dimension)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="dimension:calibration_n:test_n; repeat for multiple cases")
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    import torch

    if args.self_check:
        calibration = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64)
        test = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
        result = _one(calibration, test)
        fisher = calibration.T @ calibration / 3
        diagonal = torch.diag(torch.diag(fisher))
        assert math.isclose(result["in_sample_diagonal"], float(torch.linalg.matrix_norm(fisher - diagonal) / torch.linalg.matrix_norm(fisher)), rel_tol=1e-12)
        dimension = 3
        basis = torch.eye(dimension * dimension, dtype=torch.float64).reshape(dimension * dimension, dimension, dimension)
        columns = []
        for matrix in basis:
            transformed = torch.trace(matrix) * torch.eye(dimension, dtype=torch.float64) + matrix + matrix.T
            columns.append(transformed.reshape(-1))
        eigenvalues = torch.linalg.eigvalsh(torch.stack(columns, dim=1))
        expected = [0.0] * 3 + [2.0] * 5 + [5.0]
        assert torch.allclose(eigenvalues, torch.tensor(expected, dtype=torch.float64), atol=1e-12)
        print(json.dumps({"self_check": "ok"}))
        return
    if not args.case or not args.output:
        parser.error("--case and --output are required")
    if args.repetitions < 2:
        parser.error("--repetitions must be at least two")
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    cases = []
    for case in args.case:
        dimension, calibration_count, test_count = (int(value) for value in case.split(":"))
        if min(dimension, calibration_count, test_count) < 2:
            parser.error("all case values must be at least two")
        rows = []
        for _ in range(args.repetitions):
            calibration = torch.randn((calibration_count, dimension), generator=generator, device=device, dtype=torch.float64)
            test = torch.randn((test_count, dimension), generator=generator, device=device, dtype=torch.float64)
            rows.append(_one(calibration, test))
        cases.append({
            "dimension": dimension,
            "calibration_sample_count": calibration_count,
            "test_sample_count": test_count,
            "repetitions": args.repetitions,
            "summary": _summary(rows),
            "raw": rows,
        })
    payload = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "isotropic_gaussian_fisher_null",
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "command": [sys.executable, *sys.argv],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analytic_scaled_identity_counterexample": [_analytic_scaled_identity(dimension) for dimension in sorted({case["dimension"] for case in cases})],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "cases": len(cases), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
