# Evidence invalidation ledger

This ledger records why earlier artifacts are not evidence for the current
split-sample geometry paper. Superseded files are not part of the public
release.

## Excluded families

| Family | Reason |
|---|---|
| `r00`--`r04` and early root-level aggregates | The rank-1 Frobenius norm used `c ||mu||^4` instead of `c^2 ||mu||^4`, which could beat the oracle incorrectly. |
| `r05`--`r06` | Corrected algebra but same-sample fitting and scoring; retained only as motivation for the finite-sample null. |
| `r09` and `r11` | Unmatched or insufficiently audited GSM8K controls; replaced by the exact record/mask contract in `r12`. |
| `r13_late_dense_split` | Superseded by `r17_late_dense_matched`, which uses the same `p={0.1,0.5,0.9}`, `64|64`, and seed grid as the early dense slice. |
| MT-Bench/reversal members of `r12` | Removed from the paper because packing and source-document semantics differed; they were text-pipeline sensitivity checks, not benchmark performance. |
| All continual-learning runs (`continual_*`, `r14`, `r15`) | Outside the present paper, which makes no EWC, generation, or downstream utility claim. Earlier protocols also contain the mismatches documented in their original logs. |
| Failed LLaDA seed-0/seed-2 launches and the smoke run | Environment startup failures or development-only scale checks; the accepted retries and five complete seeds replace them. |

## Accepted evidence

- `runs/r08_split_primary/gsm_170_s*/benchmark.json`: SMDM-219M, five seeds.
- `runs/r08_split_primary/gsm_1028_s*/benchmark.json`: SMDM-1.14B, three seeds.
- `runs/r16_llada_geometry/`: accepted LLaDA-8B seeds 0--4.
- GSM8K all-target/one-target members of `runs/r12_audited_controls/`, with
  `comparison_contract.json` recomputing direct errors from sufficient
  statistics and verifying exact record/mask pairing.
- Accepted early dense members of `runs/r10_attention_split/` and the matched
  late dense members of `runs/r17_late_dense_matched/`.
- `runs/r08_split_primary/isotropic_null_full.json` and the analytic
  scaled-identity check in `experiments/simulate_fisher_null.py`.

Every accepted empirical row fits its surrogate on calibration gradients and
scores direct relative Frobenius error on disjoint examples and masks. The
submission manifest checks the complete grids, seed groups, finite values,
oracle ordering, hashes, and independent-test metadata.
