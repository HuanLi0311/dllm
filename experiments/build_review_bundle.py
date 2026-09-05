#!/usr/bin/env python3
"""Build or verify a deterministic, identity-scrubbed evidence bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1].resolve()
WORKSPACE = PROJECT_ROOT.parent
FORBIDDEN = (b"/home/", b"/Users/", b"\\Users\\")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _project_relative(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def _artifact_relative(path):
    parts = Path(path).parts
    if PROJECT_ROOT.name in parts:
        return Path(*parts[parts.index(PROJECT_ROOT.name) + 1 :])
    return Path(Path(path).name)


def _clean_string(value, host=None):
    if host and value == host:
        return "anonymous-compute-node"
    if value.startswith("/"):
        path = Path(value)
        parts = path.parts
        for anchor in (PROJECT_ROOT.name, "checkpoints"):
            if anchor in parts:
                return str(Path(*parts[parts.index(anchor) :]))
        return "python" if path.name == "python" else path.name
    return value


def _clean(value, host=None):
    if isinstance(value, dict):
        return {key: _clean(item, host) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item, host) for item in value]
    if isinstance(value, str):
        return _clean_string(value, host)
    return value


def _check_public(data, path, extra_forbidden=()):
    if any(token in data for token in (*FORBIDDEN, *extra_forbidden)):
        raise ValueError(f"identity-bearing string remains in {path}")


def _write_gzip(path, value, extra_forbidden=()):
    encoded = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    _check_public(encoded, path, extra_forbidden)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(encoded)


def build(manifest_path, output_dir):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = []
    for artifact in manifest["geometry_artifacts"]:
        source = WORKSPACE / artifact["path"]
        payload = json.loads(source.read_text(encoding="utf-8"))
        host = payload.get("host")
        cleaned = _clean(payload, host)
        cleaned["release_provenance"] = {
            "internal_path": artifact["path"],
            "internal_sha256": artifact["sha256"],
        }
        relative = _artifact_relative(artifact["path"])
        destination = output_dir / "raw" / Path(str(relative) + ".gz")
        _write_gzip(destination, cleaned, (host.encode(),) if host else ())
        raw.append({
            "internal_path": artifact["path"],
            "internal_sha256": artifact["sha256"],
            "release_path": _project_relative(destination),
            "release_sha256": _sha256(destination),
        })

    public = []
    for artifact in manifest["probes"] + manifest["extra_artifacts"]:
        path = WORKSPACE / artifact["path"]
        _check_public(path.read_bytes(), path)
        public.append({"path": artifact["path"], "sha256": artifact["sha256"]})

    manifest_copy = output_dir / "submission_manifest.json"
    manifest_data = manifest_path.read_bytes()
    _check_public(manifest_data, manifest_path)
    manifest_copy.write_bytes(manifest_data)
    result = {
        "schema_version": 2,
        "status": "ok",
        "submission_manifest": {
            "path": _project_relative(manifest_copy),
            "sha256": _sha256(manifest_copy),
        },
        "raw_artifacts": raw,
        "public_artifacts": public,
        "bundle_script_sha256": _sha256(__file__),
    }
    output = output_dir / "release_manifest.json"
    encoded = (json.dumps(result, indent=2) + "\n").encode()
    _check_public(encoded, output)
    output.write_bytes(encoded)
    return output, result


def verify(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ok":
        raise ValueError("release manifest is not successful")
    _check_public(manifest_path.read_bytes(), manifest_path)
    if _sha256(__file__) != manifest["bundle_script_sha256"]:
        raise ValueError("bundle script hash mismatch")
    listed = {artifact["release_path"] for artifact in manifest["raw_artifacts"]}
    found = {
        _project_relative(path)
        for path in (manifest_path.parent / "raw").rglob("*.gz")
    }
    if found != listed:
        raise ValueError("public raw directory does not match the release manifest")
    for artifact in manifest["raw_artifacts"]:
        path = PROJECT_ROOT / artifact["release_path"]
        if _sha256(path) != artifact["release_sha256"]:
            raise ValueError(f"release artifact hash mismatch: {path}")
        _check_public(gzip.decompress(path.read_bytes()), path)
    submission = PROJECT_ROOT / manifest["submission_manifest"]["path"]
    if _sha256(submission) != manifest["submission_manifest"]["sha256"]:
        raise ValueError("submission manifest hash mismatch")
    _check_public(submission.read_bytes(), submission)
    for artifact in manifest["public_artifacts"]:
        parts = Path(artifact["path"]).parts
        if parts and parts[0] == PROJECT_ROOT.name:
            path = PROJECT_ROOT / Path(*parts[1:])
            if _sha256(path) != artifact["sha256"]:
                raise ValueError(f"public source artifact hash mismatch: {path}")
    return {"status": "ok", "raw_artifacts": len(manifest["raw_artifacts"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evidence")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        host = "private-compute-node"
        sample = _clean({"host": host, "path": f"/home/group/user/{PROJECT_ROOT.name}/runs/x.json"}, host)
        assert sample == {"host": "anonymous-compute-node", "path": f"{PROJECT_ROOT.name}/runs/x.json"}
        print(json.dumps({"self_check": "ok"}))
        return
    if args.verify:
        print(json.dumps(verify(args.verify.resolve())))
        return
    if not args.submission_manifest:
        parser.error("--submission-manifest is required unless --verify is used")
    output, result = build(args.submission_manifest.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": "ok", "raw_artifacts": len(result["raw_artifacts"]), "output": str(output)}))


if __name__ == "__main__":
    main()
