# Public evidence

This directory is the publishable counterpart of the private `runs/` tree.
Each file under `raw/` is a deterministic gzip stream containing one
identity-scrubbed JSON experiment envelope. The numerical results are unchanged;
`release_provenance` records the SHA-256 of the corresponding internal source.

`release_manifest.json` authenticates all 25 envelopes and
`submission_manifest.json` preserves the complete internal validation record.
`comparison_contract.json` verifies exact pairing for the all-target versus
one-target intervention and recomputes both reported errors from stored
sufficient statistics.

Entries under `assets/iclr_1/` in the copied submission manifest refer to the
separately released manuscript archive; their hashes are retained for
cross-checking but the files are not duplicated in this code repository.

From the repository root:

```bash
python experiments/build_review_bundle.py --verify evidence/release_manifest.json
```

The plotting and comparison scripts read these `.json.gz` files directly; see
`../REPRODUCING.md`.
