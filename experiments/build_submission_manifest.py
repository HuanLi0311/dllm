#!/usr/bin/env python3
"""Validate the paper evidence and write a compact provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
WORKSPACE = ROOT.parent
BASE_PROBE = Path(__file__).with_name("dllm_rank1_probe.py")
LLADA_PROBE = Path(__file__).with_name("llada_geometry_probe.py")
AUDIT_PROBE = Path(__file__).with_name("run_audited_geometry_probe.py")


@lru_cache(maxsize=None)
def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(WORKSPACE))
    except ValueError:
        return resolved.name


def _csv(value, cast=str):
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _existing_file(stored, digest, fallbacks=()):
    candidates = [Path(stored), *(Path(path) for path in fallbacks)]
    for path in candidates:
        if path.is_file() and _sha256(path) == digest:
            return path.resolve()
    raise ValueError(f"no file matches stored SHA-256 {digest}: {stored}")


def _data_file(payload):
    stored = Path(payload["data"])
    name = stored.name
    return _existing_file(stored, payload["data_sha256"], (
        ROOT / "runs/data" / name,
        ROOT / "SMDM/data/gsm8k" / name,
        WORKSPACE / "runs/data" / name,
    ))


def _checkpoint(payload):
    stored = Path(payload["checkpoint"])
    if "checkpoint_files_sha256" in payload:
        for directory in (stored, WORKSPACE / "checkpoints" / stored.name):
            if directory.is_dir() and all(
                (directory / name).is_file() and _sha256(directory / name) == digest
                for name, digest in payload["checkpoint_files_sha256"].items()
            ):
                combined = hashlib.sha256(json.dumps(
                    payload["checkpoint_files_sha256"], sort_keys=True
                ).encode()).hexdigest()
                if combined != payload["checkpoint_sha256"]:
                    raise ValueError("LLaDA checkpoint aggregate hash mismatch")
                return directory.resolve()
        raise ValueError(f"LLaDA checkpoint files do not match: {stored}")
    return _existing_file(stored, payload["checkpoint_sha256"], (
        WORKSPACE / "checkpoints/mdm_safetensors" / stored.name,
        ROOT / "checkpoints" / stored.name,
    ))


def _source(payload, checkpoint):
    hashes = payload.get("source_sha256", {})
    if not hashes:
        return None
    roots = [checkpoint] if "checkpoint_files_sha256" in payload else [
        ROOT / "SMDM", Path(payload.get("code_root", ""))
    ]
    for source_root in roots:
        if source_root.is_dir() and all(
            (source_root / name).is_file() and _sha256(source_root / name) == digest
            for name, digest in hashes.items()
        ):
            return source_root.resolve()
    raise ValueError("model source files do not match the recorded hashes")


def _validate(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise ValueError(f"non-success evidence: {path}")
    checkpoint = _checkpoint(payload)
    data = _data_file(payload)
    source = _source(payload, checkpoint)
    config = payload["config"]
    sample_sizes = _csv(config["sample_sizes"], int)
    parameters = _csv(config["parameter"])
    conditions = [f"fixed_{value:g}" for value in _csv(config["mask_probabilities"], float)]
    if config.get("include_native_schedule"):
        conditions.append("native_schedule")
    expected = {
        (sample_count, parameter, condition)
        for sample_count in sample_sizes
        for parameter in parameters
        for condition in conditions
    }
    observed = {
        (row["sample_count"], row["parameter"], row["mask_condition"])
        for row in payload["results"]
    }
    if observed != expected or len(payload["results"]) != len(expected):
        raise ValueError(f"incomplete or duplicate result grid: {path}")
    for row in payload["results"]:
        if row.get("evaluation") != "split_sample":
            raise ValueError(f"non-split evaluation row: {path}")
        if row["test_sample_count"] != config["test_samples"]:
            raise ValueError(f"test size mismatch: {path}")
        if row["seed"] != config["seed"] or row["loss_mode"] != config.get("loss_mode", "native_conditional"):
            raise ValueError(f"row/config mismatch: {path}")
        numeric = [value for value in row.values() if isinstance(value, (int, float))]
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError(f"non-finite metric: {path}")
        if row["test_oracle_top1_relative_frobenius_error"] > row["mean_rank1_test_relative_frobenius_error"] + 1e-10:
            raise ValueError(f"oracle ordering violation: {path}")
    if config.get("shuffle_records") is not True:
        lines = payload.get("selected_source_lines", [])
        required = max(sample_sizes) + config["test_samples"]
        if len(lines) != required or len(set(lines)) != required:
            raise ValueError(f"independent source-document split is not recorded: {path}")
    current_hashes = {_sha256(BASE_PROBE), _sha256(LLADA_PROBE), _sha256(AUDIT_PROBE)}
    recorded_hashes = {payload.get("probe_sha256"), payload.get("base_probe_sha256")}
    if not current_hashes.intersection(recorded_hashes):
        raise ValueError(f"no current probe matches recorded code: {path}")
    return payload, {
        "path": _relative(path), "sha256": _sha256(path),
        "seed": config["seed"], "rows": len(payload["results"]),
        "checkpoint": _relative(checkpoint), "checkpoint_sha256": payload["checkpoint_sha256"],
        "data": _relative(data), "data_sha256": payload["data_sha256"],
        "source": _relative(source) if source else None,
        "probe_sha256": payload.get("probe_sha256"),
        "base_probe_sha256": payload.get("base_probe_sha256"),
    }


def build(paths, extras, contract_path=None):
    artifacts = []
    groups = defaultdict(list)
    for path in paths:
        payload, artifact = _validate(path.resolve())
        artifacts.append(artifact)
        config = payload["config"]
        key = (
            payload["checkpoint_sha256"], config["parameter"], config["sample_sizes"],
            config["mask_probabilities"], config.get("loss_mode", "native_conditional"),
        )
        groups[key].append(config["seed"])
    group_rows = []
    for key, seeds in groups.items():
        if sorted(seeds) != list(range(len(seeds))) or len(seeds) < 3:
            raise ValueError(f"evidence group needs contiguous seeds from zero: {key}, {seeds}")
        group_rows.append({
            "checkpoint_sha256": key[0], "parameter": key[1],
            "sample_sizes": key[2], "mask_probabilities": key[3],
            "loss_mode": key[4], "seeds": sorted(seeds),
        })
    contract = None
    if contract_path:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from build_comparison_contract import validate_contract
        validated = validate_contract(contract_path)
        contract = {
            "path": _relative(contract_path), "sha256": _sha256(contract_path),
            "objective_pairs": validated["objective_pairs"],
            "verified": validated["verified"],
        }
    return {
        "schema_version": 5,
        "status": "ok",
        "validated": [
            "complete Cartesian grids and contiguous seed groups",
            "finite direct errors and held-out oracle ordering",
            "disjoint calibration/test metadata",
            "data, checkpoint, source, probe, and evidence hashes",
        ],
        "probes": [
            {"path": _relative(BASE_PROBE), "sha256": _sha256(BASE_PROBE)},
            {"path": _relative(LLADA_PROBE), "sha256": _sha256(LLADA_PROBE)},
            {"path": _relative(AUDIT_PROBE), "sha256": _sha256(AUDIT_PROBE)},
        ],
        "geometry_artifacts": sorted(artifacts, key=lambda row: row["path"]),
        "matched_groups": sorted(group_rows, key=lambda row: (row["checkpoint_sha256"], row["parameter"], row["loss_mode"])),
        "comparison_contract": contract,
        "extra_artifacts": [{"path": _relative(path), "sha256": _sha256(path)} for path in extras],
        "manifest_script_sha256": _sha256(__file__),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, nargs="+")
    parser.add_argument("--extra", type=Path, nargs="*", default=[])
    parser.add_argument("--comparison-contract", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        assert _csv("8,4,8", int) == [8, 4, 8] and len(_sha256(__file__)) == 64
        print(json.dumps({"self_check": "ok"}))
        return
    if not args.geometry or not args.output:
        parser.error("--geometry and --output are required")
    result = build(args.geometry, args.extra, args.comparison_contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "artifacts": len(result["geometry_artifacts"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
