# Revised COCO measurement comparison

Historical investigation record: the corrected-only commands below belong to the externally
preserved corrected experiment snapshot, not the current publication package.

Protocol fixed on 14 September 2026, before inspecting the revised model scores.
The author authorized historical provenance documentation, explicit successful-trial
selection, a joint CV scoring correction, and renewed model comparisons. Changes
to the downstream NSD reference and the manuscript await a separate discussion.

## Historical evidence

| Experiment or artifact | Input identified | Trials | Filename image IDs | Evidence |
| --- | --- | ---: | ---: | --- |
| Paper Section 4.1 inclusion paragraph | Described as human, but counts match successful ScanDiff | 17,310 | 2,086 | Exact count match |
| Paper Tables 1 and A1, RT and joint model selection | Strong evidence for unfiltered ScanDiff, despite human captions | 31,010 | 2,779 | Recovered tables, CV denominator, commands, M1/M2 probes |
| Historical human RT reproduction | Actual human train/validation data, unsuccessful trials retained | 24,880 | 2,241 | Source JSON and model input match |
| Historical human count reproduction, Table 2 | Same human data with positive saccade count | 24,643 | 2,241 | 26 of 28 printed cells reproduced |
| Historical synthetic RT reproduction | Successful ScanDiff trials | 17,310 | 2,086 | Recorded input and manifest |
| Historical synthetic count reproduction, Table 3 | Successful ScanDiff with positive saccade count | 17,309 | 2,086 | Recorded input and reproduced table |

The recovered RT tables match 112 of 117 printed RT cells; all 27 LOO cells
match. Five printed ranking-stability cells differ slightly from the recovered
full-precision CSVs. The CV interval-overlap fractions imply a denominator of
2,779 image identities among the tested candidates. New unfiltered synthetic
M1/M2 fits closely reproduce the recovered LOO values and saved 2,779-image
rankings. This is strong input-source evidence, not a complete replay of every
historical model or fold. The original full posteriors and complete run manifest
remain missing. Historical commands alone cannot exclude stale checkpoint reuse.

The original input-source audit, recovered tables, hashes and probe results are
preserved under:

```text
/home/riccardo/data/ijcv-complexity-full-2026-09-10/rt_provenance_2026-09-14/
```

The original paper describes retaining images with at least one successful
observer. That would permit unsuccessful trials for retained images. It is not
the successful-trial rule used below. No rationale for the historical mismatch
has been established.

## Revised selection and quantities

Use the saved human and synthetic COCO JSONs, preserving source record order.
Keep each individual record only when `condition == 'present'` and `correct == 1`.
Missing correctness is excluded. Success refers to the source's correctness flag:
human response correctness and synthetic search success are not claimed to be
identical geometric measurements.

Do not add a second bounding-box filter. Of 22,946 selected human trials, 2,163
have no target-box fixation according to the existing converter. All selected
synthetic trials have one. This is recorded explicitly rather than silently
equating human response correctness with fixation containment.

| Dataset and model family | Trials | Filename image IDs | Image-target pairs | Subjects or execution indices |
| --- | ---: | ---: | ---: | ---: |
| Human, RT-only and joint | 22,946 | 2,230 | 2,474 | 10 |
| Human, count | 22,717 | 2,230 | 2,474 | 10 |
| Synthetic, RT-only and joint | 17,310 | 2,086 | 2,163 | 10 |
| Synthetic, count | 17,309 | 2,086 | 2,163 | 10 |

RT-only and joint candidates within a dataset use exactly the same valid rows.
The RT preparation requires positive RT and nonnegative N. Log-count models
require positive N, excluding 229 human and one synthetic zero-saccade trials.
No other selected rows are excluded by model validity checks in these inputs.

RT remains in milliseconds. N remains the saccade count, fixation count minus
one. Trial indices are reconstructed per subject in JSON order after success
filtering, before the model-specific validity checks. This matches the existing
synthetic conversion convention; it is not a recovered acquisition timestamp.

Filename-only model identity remains intentionally preserved. Image-target pair
counts are descriptive, not a claim that the current model fits separate effects
for targets sharing a filename. Human and synthetic supports need not match;
model selection is within each dataset. No cross-dataset ranking comparison or
change to NSD labels is part of this run.

## Models, scoring and sampling

Run all nine RT/joint models and all four count models on both datasets. Preserve
model equations, priors, units, covariates, five image-stratified subject-by-image
cell folds, and the existing ranking-stability metrics. Cells stay together;
images with fewer than two subjects are not held out by the preserved splitter.

The corrected joint CV scorer tiles each posterior scale/shape vector once per
Monte Carlo block. The concatenated predictive means use that same block order.
The former repetition paired means with another draw's scale or shape. Regression
checks compare the actual CV calculation with independently mixed scalar densities
for all four joint likelihood families. This correction does not change fits or
posterior ranking-stability calculations.

Joint LOO retains the previously restored marginalization over N. New CV uses
20 count samples per posterior draw, increased from the historical two. LOO uses
20 as before. All candidates in a dataset receive the same fold membership.
Each candidate starts its own random stream at seed 42. Fold sampling seeds are
recorded, including the stream advancement used by joint predictive simulation.

New full fits use four chains, 4,000 tuning steps and 2,000 retained draws per
chain, target acceptance 0.99, seed 42. CV uses four chains, 2,000 tuning steps
and 1,000 retained draws per chain, target acceptance 0.99, five folds. This is a
larger sampling budget, not a change to the model or an attempt to match a printed
value. Historical successful-synthetic full fits can be reused only if inputs and
scientific functions match and the expanded diagnostics pass. Their original
sampling settings and posterior hashes remain attached to the reused artifacts.

Every full fit and every CV fold is checked over all non-observation parameters,
including image effects and global loadings. Acceptance requires finite diagnostics,
zero divergences, maximum R-hat at most 1.01, minimum bulk and tail ESS at least
400, and per-chain BFMI at least 0.3. Recorded maximum-tree-depth hits must be zero.
Each comparison row carries a sampling flag. Scores from rejected fits remain
visible but cannot support a model-performance claim. PSIS-LOO warnings and Pareto
k values are recorded separately; convergence alone does not validate PSIS-LOO.

The saved successful-synthetic M2J posterior already shows two opposed chain
groups: mean RT loading about -0.1495 versus +0.1495, with between-group image
ranking correlation about -0.999. The count loading is about 0.022 in one group
and 0.0017 in the other. Thus the positive count loading does not reliably anchor
the ranking in this fit. This is observed multimodality, not an assertion of an
exact sign symmetry in the model. Increased sampling is tested without changing
priors, forcing a sign, relabelling draws, or choosing a preferred chain group.
If it fails, the result is an unresolved fitting problem, not evidence of poor
converged predictive performance.

Historical RT-scale prior-unit ambiguity and different point predictors for
log-scale RMSE remain documented in the provenance audit. They are not silently
changed here. A separate nine-candidate RT sensitivity suite uses the preserved
31,010-trial, 2,779-image unfiltered synthetic input, with the same corrected
scoring and sampling budgets. It runs after the main comparison. Its scores are
not pooled with successful-trial scores, and absolute ELPD totals across these
different observation sets are not directly comparable.

## Reproducibility and monitoring

Outputs are isolated under:

```text
/home/riccardo/data/ijcv-complexity-corrected-2026-09-14/
```

The run uses the existing Python 3.12 measurement venv at
`/home/riccardo/data/ijcv-complexity-full-2026-09-10/venvs/measurement/`.
No packages were installed. The run contains its own `requirements.freeze.txt`,
source snapshot, input hashes and counts, configuration, per-candidate commands
and logs, full posteriors or verified reuse records, and per-fold posteriors.
Fold snapshots omit unused observation-sized deterministic arrays, retaining all
parameters required to recompute predictive scores and ranking stability.

The resumable entry point is:

```bash
python -m complexity.evaluation.run_corrected_comparison --config /path/to/config.json --workers 3
```

Run it from the recorded source snapshot with the recorded venv. Reusing a job
requires exactly matching source, input, configuration and package versions.
The detached user service is `ijcv-complexity-corrected.service`. It uses CPU
sampling, with three candidates at a time and one BLAS thread per process.
On successful process exit it starts `ijcv-complexity-unfiltered.service`, whose
separate configuration, source snapshot and outputs are under `sensitivity_unfiltered/`.
The sensitivity snapshot adds an optional family selector to the scheduler;
the main run retains its already fixed source snapshot. Scientific code is identical.

```bash
systemctl --user status ijcv-complexity-corrected.service
journalctl --user -u ijcv-complexity-corrected.service -f
cat /home/riccardo/data/ijcv-complexity-corrected-2026-09-14/STATUS.md
```

Candidate logs are under `logs/`; each candidate also has a live `status.json`.
`STATUS.md` and `summary.json` are refreshed after each candidate finishes and can
be refreshed explicitly with the entry point's `--summarize` option. Completion
of computations and acceptance of sampling are reported separately. The service
survives a disconnected client and is enabled to resume after a reboot.
