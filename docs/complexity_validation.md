# Complexity pipeline validation

Validated on 9 September 2026. The full NSD M2 refit exported a CSV that is
**byte-identical to the preserved Route B ranking**. The source datasets and
historical research artifacts were read without modification.

## What changed

- Added the configurable stage runner and explicit paths for NSD correction,
  ranking export, ranked-image visualization, and ScanDiff inference.
- Enabled all four count models and all nine RT/joint models through a shared
  runner. Model equations, priors, likelihoods, preparation functions, and core
  CV calculations retain the original implementations. AST comparisons against
  both original fitting scripts confirmed that every function and class except
  the CLI entry point was unchanged at the start of the validation runs.
- Added the paper's human detectable-image selection for the human model suite.
  The initial validation also tried a human Mean-N baseline; on 10 September the
  author requested restoring the historical generated-count baseline, as recorded
  below. The existing filename-based complexity identity remains unchanged.
- Made checkpoint reuse explicit and conditional on matching source, data hash,
  packages, and settings. Failures propagate to the pipeline command.
- Added the verified variance calculation that averages repeated physical
  scene-target records before decomposition. Added regression tests for these
  scientific behaviors, without code-style tools.

## Deterministic replay

| Regenerated table | Rows | Match to historical file |
| --- | ---: | --- |
| Human COCO conversion | 24,880 | Identical bytes |
| Successful generated COCO trials | 17,310 | Identical bytes |
| Successful generated NSD trials | 116,848 | Identical bytes |
| Corrected NSD trials | 83,508 | Identical bytes |

The human detectable-image rule was evaluated using actual fixation-box
intersections, not the stored correctness flag alone. Every one of the 2,241
human image filenames has at least one qualifying observer, so this rule retains
all 24,880 target-present trials in the supplied human JSON.

Exporting the saved NSD M2 posterior produced the reference CSV with identical
bytes, labels, target mapping, scores, and order. Its SHA-256 is:

```text
2101a2143b00da45625afeaeda08880c70761f825efb0a7c17769624c0a0e6fb
```

## Full NSD M2 refit

A fresh directory was used, with checkpoint reuse disabled. The fit used the
complete corrected positive-N table: 82,221 observations and 12,447 image
filenames. The regenerated corrected table was first checked against the input
used for this fit and matched byte-for-byte.

| Setting or diagnostic | Value |
| --- | ---: |
| PyMC | 5.27.0 |
| ArviZ | 0.23.0 |
| PyTensor | 2.36.3 |
| Chains | 4 |
| Tuning steps per chain | 2,000 |
| Retained draws per chain | 2,000 |
| Seed | 42 |
| Target acceptance | 0.95 |
| Sampling time | 821.08 seconds |
| Divergences | 0 |
| Maximum image-effect R-hat, unrounded | 1.0052926792 |
| Minimum image-effect bulk ESS | 14,553 |
| Minimum image-effect tail ESS | 3,961 |

The fresh ranking is also byte-identical to the reference, with the SHA-256
above. Score RMSE and maximum absolute difference are zero, and the ordering
matches exactly. All declared reproduction criteria pass. As an additional check
that export used the fresh fit, its posterior records the new 821.08-second
sampling time, compared with 604.66 seconds for the historical artifact. The
first and last image-effect draws and all intercept and scale draws checked
against the historical posterior also matched.

Both saved-posterior replay and fresh-fit export reproduce the variance result:

| Quantity | Value |
| --- | ---: |
| Physical images with at least two targets | 1,205 |
| Distinct scene-target pairs after averaging repeats | 2,555 |
| Total variance | 0.0722349801 |
| Within-image variance | 0.0340688631 |
| Between-image variance | 0.0381661169 |
| Within-image share | 47.16394064% |

This exact reproduction was observed in this environment; future independent
runs should still be evaluated using the documented statistical criteria rather
than assuming bitwise identity.

## ScanDiff inference

The new adapter generated ten scanpaths for
`bottle/test-0004_nsd-00293_bottle.png` using the recovered visual-search checkpoint,
seed 1000, and physical GPU 2, exposed as CUDA device 0. The output contained
61 fixations in total.

For a controlled comparison, the original generator was run on the same image
in a temporary directory. Its broken plotting import/call was removed, and
checkpoint loading explicitly used `weights_only=False` for PyTorch 2.10
compatibility. The adapter's ten JSON records matched the original generator's
records exactly, including coordinates, durations, lengths, correctness, and RT.
No files in Valentyn's checkout were edited.

The [external-source manifest](../complexity/generation/source_manifest.json)
records 115 source, configuration, checkpoint, and task-embedding hashes for the
validated ScanDiff dependency. The inference environment reused the existing
PyTorch installation without changing it, while installing Hydra and timm in
an isolated environment. Public installation requirements specify the tested
versions independently of that local reuse.

## COCO model and baseline checks

### Baseline restoration, 10 September 2026

At the author's request, Mean-N again uses the successful generated COCO table,
equivalent to the historical `merged_correct.csv`. The pipeline writes it to
`rankings/predicted_mean/`, and Figure 3b compares it against human M2-RT.

The restored pipeline baseline dispatch was exercised with Bayesian fits skipped.
The complete baseline and its five-fold stability evaluation were recomputed.
The resulting 2,086-row CSV is byte-identical to
`difficulty_rankings_baseline/full_ranking_mean_N.csv`. Comparison with the saved
human M2-RT ranking gives Spearman 0.7010487569 and top-20 overlap 0.30, matching
the historical result. Human models, NSD labels, and filename grouping were not
changed. The human-baseline experiment described below is superseded for the
publication pipeline and retained only as validation history.

### Initial checks, 9 September 2026

All four count models and all nine RT/joint models completed short fits, LOO,
and two-fold cell-held-out CV on a 24-image subset. These runs used 50 tuning
steps and 50 draws for full fits, and 30 tuning steps and 30 draws for CV,
with two chains. They verify that the complete registries and evaluation paths
execute. Their sample counts are deliberately insufficient for scientific
convergence claims; their model-selection numbers are not publication results.

The human Mean-N baseline was computed on the complete selected human table
with five-fold stability evaluation. Against the saved human M2-RT ranking,
its Spearman correlation is 0.896439 and top-20 overlap is 0.50 across 2,241
filenames. The historical generated-count baseline gave 0.701049 and 0.30 across
2,086 filenames. These comparisons use their respective available image sets;
they are not a controlled estimate of the effect of changing only the estimator.

The three regression tests pass: detectable-image selection retains unsuccessful
observers, scene-target repeats are averaged before variance decomposition, and
export preserves posterior means and the historical filename-to-task behavior.
An attempted resume with changed settings was rejected. The example-image plot
also ran successfully against the prepared NSD image tree.

## Retained artifacts and scope

Run outputs are retained on this machine under:

```text
/home/riccardo/data/ijcv-complexity-2026-09-09/
```

This resolves onto the larger `/data` volume. The runs initially wrote to
`/tmp/ijcv-complexity-run/`; historical command logs and recorded paths retain
that location as provenance. The artifact directory includes the new NSD
posterior, rankings, original-versus-adapter inference outputs, smoke-test
posteriors, CV/LOO summaries, selection reports, figures, logs, and
`verification.json`. The runner used for the fit is preserved in
`source_snapshot/runner.py`; the final runner additionally records unrounded
R-hat, and that value was independently computed for this refit's final report.

The fitting and inference environments used for this session are
`/tmp/ijcv-complexity-venv/` and `/tmp/ijcv-scandiff-venv/`. They can be recreated
from the repository requirements. Large datasets and model weights remain in
their existing external locations.

Full-dataset COCO table refits and regeneration of every COCO/NSD scanpath from
images were not performed in this validation. The documented commands support
those runs. On 10 September the author confirmed four GPUs and recalled using
`--force`. The four-worker adapter and full rerun are now documented in the
[10 September reproduction record](complexity_full_reproduction.md). The prepared NSD image-construction step is
supplied as external data. The validated complete NSD replay starts from saved scanpaths;
the image-to-scanpath stage was validated separately on the controlled sample.
