#!/usr/bin/env python3
"""Generate the paper's figures directly from raw experiment envelopes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


MODEL_LABELS = {170: "SMDM-219M", 1028: "SMDM-1.14B", 8016: "LLaDA-8B"}
ROOT = Path(__file__).parents[1].resolve()
SOFT_BLUE = "#1f77b4"
SOFT_ORANGE = "#ff7f0e"
SOFT_NEUTRAL = "#f7f5f2"
PLOT_TEXT = "#173042"


def _read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def _mean_sd(values):
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _soft_diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "soft_blue_orange", [SOFT_ORANGE, SOFT_NEUTRAL, SOFT_BLUE]
    )


def _cell_text(value):
    return f"{0.0 if abs(value) < 0.005 else value:+.2f}"


def _layer(parameter):
    match = re.search(r"(?:\.h\.|\.blocks\.)(\d+)\.", parameter)
    if not match:
        raise ValueError(f"cannot parse layer from parameter name: {parameter}")
    return int(match.group(1))


def _corpus(data):
    path = Path(data)
    stem = path.stem
    if stem == "gsm8k_tasks" or "gsm8k" in {part.lower() for part in path.parts}:
        return "GSM8K"
    return stem


def _rank1_error(record):
    return record.get("mean_rank1_test_relative_frobenius_error", record.get("rank1_relative_frobenius_error"))


def _diagonal_error(record):
    return record.get("diagonal_test_relative_frobenius_error", record.get("diagonal_relative_frobenius_error"))


def _load_geometry(paths):
    records, files, failures = [], [], []
    for path in paths:
        payload = _read_json(path)
        files.append({"path": _relative(path), "sha256": _sha256(path), "status": payload.get("status")})
        if payload.get("status") != "ok":
            failures.append({"path": _relative(path), "error": payload.get("error")})
            continue
        corpus = _corpus(payload["data"])
        for record in payload["results"]:
            records.append({**record, "corpus": corpus, "source": _relative(path)})
    if not records:
        raise ValueError("no successful geometry records")
    violations = []
    for record in records:
        oracle = record.get("test_oracle_top1_relative_frobenius_error", record.get("oracle_top1_relative_frobenius_error"))
        if oracle > _rank1_error(record) + 1e-10:
            violations.append(record)
    if violations:
        raise ValueError(f"oracle ordering violated in {len(violations)} records")
    return records, files, failures


def _groups(records, keys):
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    return grouped


def _save(fig, output_dir, stem):
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"Creator": "experiments/make_paper_figures.py", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")


def _model_label(model):
    return MODEL_LABELS.get(model, str(model))


def _difference(record):
    """Positive values mean lower independent-test error for mean-gradient rank-1."""
    return _diagonal_error(record) - _rank1_error(record)


def _heatmaps(plt, np, records, corpus, output_dir):
    selected = [record for record in records if record["corpus"] == corpus and record["mask_probability"] is not None and record.get("loss_mode") in (None, "native_conditional")]
    models = sorted({record["model_size_m"] for record in selected})
    cells, arrays = [], []
    for model in models:
        model_records = [record for record in selected if record["model_size_m"] == model]
        sample_count = max(record["sample_count"] for record in model_records)
        parameters = sorted({record["parameter"] for record in model_records}, key=_layer)
        probabilities = sorted({record["mask_probability"] for record in model_records})
        grouped = _groups(
            [record for record in model_records if record["sample_count"] == sample_count],
            ("parameter", "mask_probability"),
        )
        array = []
        for parameter in parameters:
            row = []
            for probability in probabilities:
                values = grouped[(parameter, probability)]
                delta = statistics.fmean(_difference(record) for record in values)
                row.append(delta)
                cells.append({"corpus": corpus, "model_size_m": model, "sample_count": sample_count, "layer": _layer(parameter), "mask_probability": probability, "mean_diagonal_minus_rank1_error": delta, "seeds": sorted(record["seed"] for record in values)})
            array.append(row)
        arrays.append((model, sample_count, parameters, probabilities, np.array(array)))
    fig, axes = plt.subplots(1, len(arrays), figsize=(4.35 * len(arrays), 3.7), constrained_layout=True, squeeze=False)
    cmap = _soft_diverging_cmap()
    for panel, (model, sample_count, parameters, probabilities, array) in zip(axes[0], arrays):
        # ponytail: independent panel scales keep finite SMDM differences readable
        # beside the heavy-tailed LLaDA estimate; annotated cells retain exact values.
        vmax = max(abs(float(array.min())), abs(float(array.max())), 1e-12)
        image = panel.imshow(array, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        panel.set_xticks(range(len(probabilities)), [f"{value:.1f}" for value in probabilities])
        panel.set_yticks(range(len(parameters)), [f"layer {_layer(value)}" for value in parameters])
        panel.set_title(f"{_model_label(model)}, n={sample_count}", fontsize=13)
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                value = array[row, column]
                panel.text(column, row, _cell_text(value), ha="center", va="center", fontsize=9, color=PLOT_TEXT)
        colorbar = fig.colorbar(image, ax=panel, shrink=0.82)
        colorbar.ax.set_title(r"$\Delta$", fontsize=11)
    axes[0, 0].set_ylabel("parameter slice")
    fig.supxlabel("mask probability")
    _save(fig, output_dir, "geometry_heatmaps")
    plt.close(fig)
    return cells


def _robustness(plt, np, records, corpus, output_dir):
    selected = [record for record in records if record["corpus"] == corpus and record["mask_probability"] is not None and record.get("loss_mode") in (None, "native_conditional")]
    models = sorted({record["model_size_m"] for record in selected})
    rows = []
    fig, axes = plt.subplots(1, len(models), figsize=(3.7 * len(models), 3.6), constrained_layout=True, squeeze=False)
    for panel, model in zip(axes[0], models):
        model_records = [record for record in selected if record["model_size_m"] == model]
        sample_sizes = sorted({record["sample_count"] for record in model_records})
        method_values = {"rank-1": ([], []), "diagonal": ([], [])}
        for sample_count in sample_sizes:
            subset = [record for record in model_records if record["sample_count"] == sample_count]
            by_seed = _groups(subset, ("seed",))
            seed_rank1 = [statistics.fmean(_rank1_error(record) for record in values) for values in by_seed.values()]
            seed_diagonal = [statistics.fmean(_diagonal_error(record) for record in values) for values in by_seed.values()]
            for label, values in (("rank-1", seed_rank1), ("diagonal", seed_diagonal)):
                mean, sd = _mean_sd(values)
                method_values[label][0].append(mean)
                method_values[label][1].append(sd)
            rows.append({
                "corpus": corpus,
                "model_size_m": model,
                "sample_count": sample_count,
                "rank1_seed_means": seed_rank1,
                "rank1_mean": statistics.fmean(seed_rank1),
                "rank1_sd": statistics.stdev(seed_rank1),
                "diagonal_seed_means": seed_diagonal,
                "diagonal_mean": statistics.fmean(seed_diagonal),
                "diagonal_sd": statistics.stdev(seed_diagonal),
            })
        for label, color in (("rank-1", SOFT_BLUE), ("diagonal", SOFT_ORANGE)):
            means, sds = method_values[label]
            panel.errorbar(sample_sizes, means, yerr=sds, marker="o", capsize=4, color=color, label=label)
        panel.set_xscale("log", base=2)
        panel.set_xlabel("calibration examples")
        panel.set_title(_model_label(model))
        panel.axhline(1.0, color="0.55", linewidth=1, linestyle=":")
    axes[0, 0].set_ylabel("held-out relative Frobenius error\n(mean $\\pm$ SD across seeds)")
    axes[0, 0].legend(frameon=False)
    _save(fig, output_dir, "geometry_robustness")
    plt.close(fig)
    return rows


def _objective_heatmaps(plt, np, records, output_dir):
    selected = [record for record in records if record["corpus"] == "GSM8K" and record["model_size_m"] == 170 and record["mask_probability"] is not None and record.get("evaluation") == "split_sample"]
    modes = [mode for mode in ("native_conditional", "fixed_target") if any(record.get("loss_mode") == mode for record in selected)]
    if len(modes) < 2:
        return []
    common_seeds = set.intersection(*[{record["seed"] for record in selected if record.get("loss_mode") == mode} for mode in modes])
    if not common_seeds:
        raise ValueError("target-count comparison has no common seeds")
    labels = {"native_conditional": "all targets", "fixed_target": "one target"}
    arrays, rows, mode_arrays = [], [], {}
    for mode in modes:
        subset = [record for record in selected if record["loss_mode"] == mode and record["sample_count"] == 64 and record["seed"] in common_seeds]
        parameters = sorted({record["parameter"] for record in subset}, key=_layer)
        probabilities = sorted({record["mask_probability"] for record in subset})
        grouped = _groups(subset, ("parameter", "mask_probability"))
        array = np.array([[statistics.fmean(_difference(record) for record in grouped[(parameter, probability)]) for probability in probabilities] for parameter in parameters])
        arrays.append((labels[mode], parameters, probabilities, array))
        mode_arrays[mode] = array
        for row_index, parameter in enumerate(parameters):
            for column, probability in enumerate(probabilities):
                rows.append({"loss_mode": mode, "sample_count": 64, "layer": _layer(parameter), "mask_probability": probability, "mean_diagonal_minus_rank1_error": float(array[row_index, column]), "seeds": sorted(common_seeds)})
    effect = mode_arrays["native_conditional"] - mode_arrays["fixed_target"]
    arrays.append((r"paired margin change", parameters, probabilities, effect))
    for row_index, parameter in enumerate(parameters):
        for column, probability in enumerate(probabilities):
            rows.append({"loss_mode": "all_minus_one_target", "sample_count": 64, "layer": _layer(parameter), "mask_probability": probability, "mean_error_margin_change": float(effect[row_index, column]), "seeds": sorted(common_seeds)})
    vmax = max(max(abs(float(array.min())), abs(float(array.max()))) for *_, array in arrays)
    fig, axes = plt.subplots(1, len(arrays), figsize=(10.8, 3.8), constrained_layout=True, squeeze=False)
    image = None
    cmap = _soft_diverging_cmap()
    for panel, (label, parameters, probabilities, array) in zip(axes[0], arrays):
        image = panel.imshow(array, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        panel.set_xticks(range(len(probabilities)), [f"{value:.1f}" for value in probabilities])
        panel.set_yticks(range(len(parameters)), [f"L{_layer(value)}" for value in parameters])
        panel.set_title(label)
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                value = array[row, column]
                panel.text(column, row, _cell_text(value), ha="center", va="center", fontsize=11, color=PLOT_TEXT)
    axes[0, 0].set_ylabel("219M parameter slice")
    fig.supxlabel("context mask probability")
    fig.colorbar(image, ax=axes, shrink=0.85, label=r"$e_{\mathrm{diag}}-e_{\mathrm{rank1}}$")
    _save(fig, output_dir, "objective_heatmaps")
    plt.close(fig)
    return rows


def _real_in_vs_out(plt, records, corpus, output_dir):
    selected = [record for record in records if record["corpus"] == corpus and record.get("evaluation") == "split_sample" and record.get("loss_mode") == "native_conditional" and record["mask_probability"] is not None]
    if not selected:
        return []
    models = sorted({record["model_size_m"] for record in selected})
    fig, axes = plt.subplots(1, len(models), figsize=(3.7 * len(models), 3.7), constrained_layout=True, squeeze=False)
    rows = []
    for panel, model in zip(axes[0], models):
        model_records = [record for record in selected if record["model_size_m"] == model]
        sample_sizes = sorted({record["sample_count"] for record in model_records})
        curves = {
            "rank-1, calibration": ([], []),
            "diagonal, calibration": ([], []),
            "rank-1, held out": ([], []),
            "diagonal, held out": ([], []),
        }
        for sample_count in sample_sizes:
            subset = [record for record in model_records if record["sample_count"] == sample_count]
            by_seed = _groups(subset, ("seed",))
            definitions = {
                "rank-1, calibration": "calibration_rank1_relative_frobenius_error",
                "diagonal, calibration": "calibration_diagonal_relative_frobenius_error",
                "rank-1, held out": "mean_rank1_test_relative_frobenius_error",
                "diagonal, held out": "diagonal_test_relative_frobenius_error",
            }
            row = {"corpus": corpus, "model_size_m": model, "sample_count": sample_count}
            for label, key in definitions.items():
                values = [statistics.fmean(record[key] for record in seed_records) for seed_records in by_seed.values()]
                mean, sd = _mean_sd(values)
                curves[label][0].append(mean)
                curves[label][1].append(sd)
                row[label] = {"seed_means": values, "mean": mean, "sd": sd}
            rows.append(row)
        for label, color, linestyle in (
            ("rank-1, calibration", SOFT_BLUE, "-"),
            ("diagonal, calibration", SOFT_ORANGE, "-"),
            ("rank-1, held out", SOFT_BLUE, "--"),
            ("diagonal, held out", SOFT_ORANGE, "--"),
        ):
            means, sds = curves[label]
            panel.errorbar(sample_sizes, means, yerr=sds, marker="o", capsize=3, color=color, linestyle=linestyle, label=label)
        panel.set_xscale("log", base=2)
        panel.set_xlabel("calibration examples")
        panel.set_title(_model_label(model))
        panel.axhline(1.0, color="0.55", linewidth=1, linestyle=":")
    axes[0, 0].set_ylabel("relative Frobenius error\n(mean $\\pm$ SD across seeds)")
    axes[0, 0].legend(fontsize=9, frameon=False)
    _save(fig, output_dir, "real_in_sample_vs_test")
    plt.close(fig)
    return rows


def _null_figures(plt, np, null_path, output_dir):
    if not null_path:
        return {}
    payload = _read_json(null_path)
    dimensions = sorted({case["dimension"] for case in payload["cases"]})
    colors = {dimensions[0]: "#1f77b4", dimensions[-1]: "#ff7f0e"}
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), constrained_layout=True)
    for panel, prefix, title in zip(axes, ("in_sample", "split", "population"), ("same-sample score", "independent-test score", "true population score")):
        for dimension in dimensions:
            cases = sorted((case for case in payload["cases"] if case["dimension"] == dimension), key=lambda case: case["calibration_sample_count"])
            x_values = np.array([case["calibration_sample_count"] / dimension for case in cases])
            y_values = np.array([case["summary"][f"{prefix}_rank1_win_rate"] for case in cases])
            panel.plot(
                x_values,
                y_values,
                marker="o",
                color=colors[dimension],
                label=f"d={dimension}",
            )
        panel.axhline(0.5, color="0.5", linestyle="--", linewidth=1.2)
        panel.set_xscale("log", base=2)
        panel.set_ylim(-0.03, 1.03)
        panel.set_xlabel("calibration ratio n/d")
        panel.set_title(title)
    axes[0].set_ylabel("P(rank-1 error < diagonal error)")
    axes[-1].legend(frameon=False)
    _save(fig, output_dir, "finite_sample_null")
    plt.close(fig)

    dimensions_curve = np.unique(np.geomspace(2, 4096, 300).astype(int))
    eigengap = 2 / (dimensions_curve + 2)
    rank1_error = np.sqrt(2 * (dimensions_curve - 1) / (3 * dimensions_curve))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    # ponytail: these are exact analytic curves, so no sampling interval is meaningful.
    axes[0].plot(dimensions_curve, eigengap, color=SOFT_BLUE)
    axes[1].plot(dimensions_curve, rank1_error, color=SOFT_ORANGE)
    for panel in axes:
        panel.set_xscale("log")
        panel.set_xlabel("data dimension d")
    axes[0].set_ylabel(r"$\lambda_2/\lambda_1=2/(d+2)$")
    axes[0].set_title("spectral ratio vanishes")
    axes[1].set_ylabel("best rank-1 Frobenius error")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(math.sqrt(2 / 3), color="0.45", linewidth=1.2, linestyle=":")
    axes[1].text(
        0.98,
        math.sqrt(2 / 3) + 0.015,
        r"$\sqrt{2/3}$ limit",
        transform=axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=12,
        color="0.35",
    )
    axes[1].set_title("rank-1 residual stays large")
    for label, panel in zip("AB", axes):
        panel.text(-0.14, 1.05, label, transform=panel.transAxes, fontweight="bold")
    _save(fig, output_dir, "gaussian_counterexample")
    plt.close(fig)
    return {
        "path": _relative(null_path),
        "sha256": _sha256(null_path),
        "cases": [{"dimension": case["dimension"], "calibration_sample_count": case["calibration_sample_count"], "test_sample_count": case["test_sample_count"], "summary": case["summary"]} for case in payload["cases"]],
        "analytic_scaled_identity_counterexample": payload["analytic_scaled_identity_counterexample"],
    }


def _score_summary(records):
    by_seed = _groups(records, ("seed",))
    rank1_seed_means = [statistics.fmean(_rank1_error(record) for record in by_seed[key]) for key in sorted(by_seed)]
    diagonal_seed_means = [statistics.fmean(_diagonal_error(record) for record in by_seed[key]) for key in sorted(by_seed)]
    oracle_seed_means = [statistics.fmean(record["test_oracle_top1_relative_frobenius_error"] for record in by_seed[key]) for key in sorted(by_seed)]
    cells = _groups(records, ("parameter", "mask_condition"))
    cell_differences = [statistics.fmean(_difference(record) for record in values) for values in cells.values()]
    return {
        "records": len(records),
        "seeds": sorted(key[0] for key in by_seed),
        "conditions": len(cells),
        "rank1_seed_mean_errors": rank1_seed_means,
        "rank1_mean_error": statistics.fmean(rank1_seed_means),
        "rank1_sd": statistics.stdev(rank1_seed_means),
        "diagonal_seed_mean_errors": diagonal_seed_means,
        "diagonal_mean_error": statistics.fmean(diagonal_seed_means),
        "diagonal_sd": statistics.stdev(diagonal_seed_means),
        "oracle_rank1_seed_mean_errors": oracle_seed_means,
        "oracle_rank1_mean_error": statistics.fmean(oracle_seed_means),
        "oracle_rank1_sd": statistics.stdev(oracle_seed_means),
        "cell_win_count": sum(value > 0 for value in cell_differences),
        "cell_count": len(cell_differences),
    }


def _submission_summary(records, comparison_records, corpus):
    native = [record for record in records if record["corpus"] == corpus and record.get("evaluation") == "split_sample" and record.get("loss_mode") == "native_conditional"]
    primary = []
    for model in sorted({record["model_size_m"] for record in native}):
        model_records = [record for record in native if record["model_size_m"] == model]
        sample_count = max(record["sample_count"] for record in model_records)
        fixed = [record for record in model_records if record["sample_count"] == sample_count and record["mask_probability"] is not None]
        primary.append({
            "model_size_m": model,
            "parameter_count_label": _model_label(model),
            "calibration_sample_count": sample_count,
            "fixed_mask_grid": _score_summary(fixed),
        })

    modes = {}
    for index, mode in enumerate(("native_conditional", "fixed_target")):
        subset = [record for record in comparison_records if record["corpus"] == corpus and record["model_size_m"] == 170 and record.get("loss_mode") == mode and record["sample_count"] == 64 and record["mask_probability"] is not None and record["seed"] in {0, 1, 2}]
        modes[mode] = _score_summary(subset)
    all_rows = [record for record in comparison_records if record["corpus"] == corpus and record["model_size_m"] == 170 and record.get("loss_mode") == "native_conditional" and record["sample_count"] == 64 and record["mask_probability"] is not None and record["seed"] in {0, 1, 2}]
    one_rows = [record for record in comparison_records if record["corpus"] == corpus and record["model_size_m"] == 170 and record.get("loss_mode") == "fixed_target" and record["sample_count"] == 64 and record["mask_probability"] is not None and record["seed"] in {0, 1, 2}]
    all_lookup = {(record["seed"], record["parameter"], record["mask_probability"]): _difference(record) for record in all_rows}
    one_lookup = {(record["seed"], record["parameter"], record["mask_probability"]): _difference(record) for record in one_rows}
    if all_lookup.keys() != one_lookup.keys():
        raise ValueError("target-count intervention is not paired")
    differences = {key: all_lookup[key] - one_lookup[key] for key in all_lookup}
    seed_effects = [statistics.fmean(value for (run_seed, _, _), value in differences.items() if run_seed == seed) for seed in sorted({key[0] for key in differences})]
    effect_mean, effect_sd = _mean_sd(seed_effects)
    cell_effects = _groups([{"parameter": parameter, "mask_probability": probability, "effect": value} for (_, parameter, probability), value in differences.items()], ("parameter", "mask_probability"))
    cell_effect_values = [statistics.fmean(row["effect"] for row in values) for values in cell_effects.values()]
    target_effect = {
        "seed_mean_margin_changes": seed_effects,
        "mean_margin_change": effect_mean,
        "sd": effect_sd,
        "positive_cell_count": sum(value > 0 for value in cell_effect_values),
        "cell_count": len(cell_effect_values),
    }
    return {"primary": primary, "target_count_modes": modes, "paired_target_aggregation_effect": target_effect}


def _dense_slice_control(plt, np, records, output_dir):
    if not records:
        return {}
    selected = [
        record for record in records
        if record.get("evaluation") == "split_sample"
        and record.get("loss_mode") == "native_conditional"
        and ".attn.proj.weight" in record["parameter"]
        and record["mask_probability"] is not None
    ]
    sample_count = max(record["sample_count"] for record in selected)
    selected = [record for record in selected if record["sample_count"] == sample_count]
    parameters = sorted({record["parameter"] for record in selected}, key=_layer)
    fig, axes = plt.subplots(1, len(parameters), figsize=(4.3 * len(parameters), 3.6), constrained_layout=True, squeeze=False)
    rows = []
    for panel, parameter in zip(axes[0], parameters):
        subset = [record for record in selected if record["parameter"] == parameter]
        probabilities = sorted({record["mask_probability"] for record in subset})
        grouped = _groups(subset, ("mask_probability",))
        rank1_means, rank1_sds, diagonal_means, diagonal_sds = [], [], [], []
        for probability in probabilities:
            values = grouped[(probability,)]
            rank1 = [_rank1_error(record) for record in values]
            diagonal = [_diagonal_error(record) for record in values]
            rank1_mean, rank1_sd = _mean_sd(rank1)
            diagonal_mean, diagonal_sd = _mean_sd(diagonal)
            rank1_means.append(rank1_mean)
            rank1_sds.append(rank1_sd)
            diagonal_means.append(diagonal_mean)
            diagonal_sds.append(diagonal_sd)
            rows.append({
                "parameter": parameter,
                "layer": _layer(parameter),
                "mask_probability": probability,
                "rank1_seed_errors": rank1,
                "rank1_mean": rank1_mean,
                "rank1_sd": rank1_sd,
                "diagonal_seed_errors": diagonal,
                "diagonal_mean": diagonal_mean,
                "diagonal_sd": diagonal_sd,
            })
        panel.errorbar(probabilities, rank1_means, yerr=rank1_sds, marker="o", capsize=4, color=SOFT_BLUE, label="rank-1")
        panel.errorbar(probabilities, diagonal_means, yerr=diagonal_sds, marker="o", capsize=4, color=SOFT_ORANGE, label="diagonal")
        panel.axhline(1.0, color="0.45", linewidth=1.2, linestyle=":")
        panel.set_xlabel("mask probability")
        panel.set_title(f"layer {_layer(parameter)} attention output")
    axes[0, 0].set_ylabel("held-out relative Frobenius error")
    axes[0, 0].legend(frameon=False)
    _save(fig, output_dir, "dense_slice_control")
    plt.close(fig)
    return {"sample_count": sample_count, "test_sample_count": selected[0]["test_sample_count"], "cells": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, nargs="+")
    parser.add_argument("--comparison-control", type=Path, nargs="+")
    parser.add_argument("--comparison-contract", type=Path)
    parser.add_argument("--slice-control", type=Path, nargs="+")
    parser.add_argument("--null", type=Path)
    parser.add_argument("--primary-corpus", default="GSM8K")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        import numpy as np

        assert _mean_sd([1.0, 2.0, 3.0]) == (2.0, 1.0)
        assert _layer("transformer.h.17.norm_1.weight") == 17
        assert _read_json(Path(__file__).parents[1] / "runs/data/gsm8k_tasks_manifest.json")["schema_version"] == 1
        rows = [
            {"seed": seed, "parameter": f"transformer.h.{layer}.norm_1.weight", "mask_probability": 0.5, "mask_condition": "fixed_0.5", "diagonal_relative_frobenius_error": 1.0, "rank1_relative_frobenius_error": value, "test_oracle_top1_relative_frobenius_error": value / 2}
            for seed, layer, value in ((0, 0, 0.7), (0, 1, 0.8), (1, 0, 0.6), (1, 1, 0.9))
        ]
        summary = _score_summary(rows)
        assert summary["records"] == 4 and summary["conditions"] == 2
        print(json.dumps({"self_check": "ok"}))
        return
    if not args.geometry or not args.comparison_control or not args.comparison_contract:
        parser.error("--geometry, --comparison-control, and --comparison-contract are required")

    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, files, failures = _load_geometry(args.geometry)
    comparison_records, comparison_files, comparison_failures = _load_geometry(args.comparison_control)
    from build_comparison_contract import validate_contract
    comparison_contract = validate_contract(args.comparison_contract)
    if args.slice_control:
        slice_records, slice_files, slice_failures = _load_geometry(args.slice_control)
    else:
        slice_records, slice_files, slice_failures = [], [], []
    heatmaps = _heatmaps(plt, np, records, args.primary_corpus, args.output_dir)
    sensitivity = _robustness(plt, np, records, args.primary_corpus, args.output_dir)
    split_mode = any(record.get("evaluation") == "split_sample" for record in records)
    real_in_vs_out = _real_in_vs_out(plt, records, args.primary_corpus, args.output_dir) if split_mode else []
    objective = _objective_heatmaps(plt, np, comparison_records, args.output_dir)
    null = _null_figures(plt, np, args.null, args.output_dir)
    submission_summary = _submission_summary(records, comparison_records, args.primary_corpus) if split_mode else {}
    dense_slice_control = _dense_slice_control(plt, np, slice_records, args.output_dir)
    figure_data = {
        "schema_version": 3,
        "figure_script_sha256": _sha256(Path(__file__)),
        "geometry_inputs": files,
        "geometry_failures": failures,
        "comparison_control_inputs": comparison_files,
        "comparison_control_failures": comparison_failures,
        "comparison_contract": {
            "path": _relative(args.comparison_contract),
            "sha256": _sha256(args.comparison_contract),
            "verified": comparison_contract["verified"],
        },
        "slice_control_inputs": slice_files,
        "slice_control_failures": slice_failures,
        "primary_corpus": args.primary_corpus,
        "heatmap_cells": heatmaps,
        "sample_sensitivity": sensitivity,
        "real_in_sample_vs_test": real_in_vs_out,
        "objective_cells": objective,
        "null": null,
        "submission_summary": submission_summary,
        "dense_slice_control": dense_slice_control,
    }
    (args.output_dir / "figure_data.json").write_text(json.dumps(figure_data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "geometry_records": len(records), "figures": sorted(path.name for path in args.output_dir.glob("*.pdf"))}, indent=2))


if __name__ == "__main__":
    main()
