# Project guidance

- Keep the scientific claim narrow: this repository measures held-out Fisher
  geometry; it does not establish downstream EWC or task-performance gains.
- Treat `runs/` as private local provenance. Only `runs/data/` is publishable;
  public experiment envelopes belong in `evidence/` after sanitization.
- Do not edit accepted evidence by hand. Change a probe, rerun it, then rebuild
  the comparison contract, figures, submission manifest, and public bundle.
- Run every `--self-check` command in `README.md` after changing non-trivial
  logic. Run the evidence verifier before publishing.
- Never commit checkpoints, credentials, absolute user paths, or compute-node
  names.
