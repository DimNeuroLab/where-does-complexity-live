# Full complexity reproduction, 10 September 2026

For the subsequent RT provenance findings, successful-trial protocol, and corrected
CV investigation, see the [revised comparison](complexity_corrected_protocol.md).
The reproduction results below describe the earlier historical run.

Updated 14 September 2026: generation, all model fits, LOO, and five-fold CV
finished on 12 September. After a machine restart, the supervisor repeated the
final exports and checks on 14 September. Fresh NSD ranking verification failed
because 14 labels were added and 13 were missing. Execution completion does not
establish reproduction of all paper results or convergence of every model.

The [discrepancy investigation](complexity_discrepancies.md) records the filter
mechanism behind those label changes, table comparisons, diagnostic failures,
and a corrected joint-model LOO migration omission. The original run outputs and
source snapshot remain unchanged. The earlier NSD refit from saved scanpaths
still reproduces the Route B ranking exactly, as recorded in
[validation](complexity_validation.md).

## Environments and artifacts

All new large outputs and both isolated Python 3.12 environments are under:

```text
/home/riccardo/data/ijcv-complexity-full-2026-09-10/
```

This path resolves onto `/data/riccardo/`. Neither environment inherits packages
from another project's environment or from system site-packages.

The environments were created with `python3.12 -m venv` and installed with their
own `python -m pip`. From the repository root, equivalent setup commands are:

```bash
export RUN_ROOT=/home/riccardo/data/ijcv-complexity-full-2026-09-10
python3.12 -m venv "$RUN_ROOT/venvs/measurement"
python3.12 -m venv "$RUN_ROOT/venvs/scandiff"
PIP_CACHE_DIR="$RUN_ROOT/wheel-cache" \
  "$RUN_ROOT/venvs/measurement/bin/python" -m pip install -r complexity/requirements.txt
PIP_CACHE_DIR="$RUN_ROOT/wheel-cache" \
  "$RUN_ROOT/venvs/scandiff/bin/python" -m pip install -r complexity/generation/requirements.lock.txt
"$RUN_ROOT/venvs/measurement/bin/python" -m pip check
"$RUN_ROOT/venvs/scandiff/bin/python" -m pip check
```

The actual initial ScanDiff installation used `generation/requirements.txt`;
the resulting complete freeze is now [requirements.lock.txt](../complexity/generation/requirements.lock.txt).
The measurement freeze exactly matches [requirements.txt](../complexity/requirements.txt).
Both `pip check` commands passed. Install logs, complete freezes, check results,
and system details are retained in `environments/`. Main versions are PyMC 5.27.0,
ArviZ 0.23.0, Torch 2.10.0, torchvision 0.25.0, and timm 1.0.9.

Hardware is four NVIDIA L40S GPUs, each with about 46 GB, driver 580.173.02.
GPU 0 was occupied. NSD's four logical workers use physical devices
`cuda:1`, `cuda:2`, `cuda:3`, and `cuda:1`; COCO uses `cuda:2`.
Generation for the two datasets runs concurrently. Sharing devices affects
throughput; each process retains its own random stream.

`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MPLBACKEND=Agg` are recorded
for the fitting jobs. ScanDiff uses the first two settings. `inputs_manifest.json`
and the two image manifests record input hashes, including the paper, full NSD
metadata, saved scanpaths, prepared images, and cached DINO weights. The existing
external ScanDiff source manifest records the model source and checkpoint hashes.
`source_snapshot/` preserves the code used by generation and downstream jobs.

## Recovered inference settings

The author's recollection was four GPUs and `--force`. Artifact inspection
supports different settings for COCO and NSD:

| Dataset | Worker schedule | Precision | Evidence |
| --- | --- | --- | --- |
| COCO | One sorted stream, seed 1000 | float32 | All 3,101 historical output modification times follow strict filename order, about eight seconds apart, spanning 24,903 seconds. The first two full-precision inputs closely match historical records. |
| NSD | Four contiguous chunks, seeds 1000 through 1003 | Mixed precision | The earliest output times interleave the starts of the four recovered partitions. Saved coordinates exhibit half-precision quantization, consistent with the recovered generator's autocast context. |

These are evidence-based reproduction settings. The historical launch command,
original environment, and exact arithmetic configuration were not recovered.
COCO's four-worker exploratory check showed large differences outside the first
worker, supporting the single-stream interpretation. For the first two COCO
inputs in that stream, all 20 scanpath lengths and success flags match; the
largest coordinate difference is about 0.0011 pixels and the largest total RT
difference is about 0.00028 milliseconds. Full precision is substantially closer
to these historical outputs than mixed precision. This sample does not establish
identity across the complete dataset. The completed comparison subsequently
confirmed identical fixation lengths and success flags for all 31,010 COCO
records, with small coordinate and duration differences. See the investigation
for the full comparison and the different NSD outcome.

The example config records four mixed-precision workers as the default and
overrides COCO with one float32 worker. `generation.profiles` contains these
dataset overrides. The private run config records the physical GPU mapping.

### Controlled implementation checks

- Eight valid NSD inputs distributed over four workers produced 80 records
  exactly equal to the recovered original generator run in the same environment.
- A second check included the same eight valid inputs and all five unreadable
  NSD files. Both implementations produced the same 80 records and skipped the
  same five inputs. Restoring worker 0's random state from after its first image
  and recomputing its remaining images reproduced the saved records exactly.
- These comparisons removed the original broken plotting call, explicitly set
  `weights_only=False` for checkpoint loading, and mapped the original four
  worker processes to the same physical devices as the adapter.
- The recovered mixed-precision implementation does not exactly reproduce the
  saved historical JSONs in this environment. Controlled implementation equality
  and historical data equality are separate checks.

Evidence is retained under `four_worker_check/`, `four_worker_corrupt_check/`,
and `full_precision_stream_check/`. The initial partial mixed-precision COCO run
was stopped and preserved under `recovered_amp_partial/`; it is not used as a
complete publication dataset. Its source snapshot and launch records are retained.

The five unreadable NSD inputs are:

```text
bottle/train-3512_nsd-26119_bottle.png
chair/train-3512_nsd-26119_chair.png
clock/train-4708_nsd-34850.png
clock/train-5510_nsd-41056.png
knife/train-8576_nsd-63944.png
```

Their decoding errors explain the 52,486 prepared inputs versus 52,481 historical
outputs. They remain in the sorted partition list and are skipped before any
diffusion random draws, matching the original error path.

## Completed work and automatic continuation

| Branch | Work |
| --- | --- |
| `historical/` | All four COCO suites from saved scanpaths: human and generated, count and RT/joint. Four suites run concurrently. |
| `regenerated/` | All COCO and NSD scanpaths from prepared images, followed by trial preparation, generated COCO suites, and the NSD M2 fit. |
| Comparisons | Tables 1, 2, 3, and A1; Figures 3 and 4 ranking comparisons; historical versus generated scanpaths; NSD variance and final Route B ranking verification. |

Each COCO branch includes four count models and nine RT/joint models for each
data source. Full fits use four chains, 2,000 tuning steps, 2,000 retained draws,
seed 42, and target acceptance 0.95. Five-fold CV uses two chains, 600 tuning
steps, and 600 retained draws per fold. These CV settings are explicit rerun
settings, not confirmed historical settings. Mean-N uses generated COCO trials.
Historical filename grouping is preserved, including the deferred COCO issue.

Human fits are shared between branches only after checking that the human input
tables are byte-identical. Generated COCO fits are recomputed. NSD M2 is fitted
to the regenerated and corrected scanpaths, exported, and verified against the
preserved Route B reference. A failed verification remains a failed result.

Machine-specific launch and continuation scripts are retained in the artifact
directory as `ijcv_launch_full_fits.py`, `ijcv_generate_full.py`, and
`ijcv_finish_full.py`. These are the initial launch records. The independent
supervisor described below now controls continuation. Exact initial commands are
in `fit_commands.json`, `generation_status.json`, and `continuation_logs/`;
new commands are retained in `supervisor_logs/`.
Individual model manifests and worker `progress.json` files provide finer progress.

Inspect the overall state with:

```bash
cat "$RUN_ROOT/generation_status.json"
cat "$RUN_ROOT/supervisor_status.json"
```

`full_run_status.json` is written after all three continuation branches finish.
It records execution outcomes; consult the numerical reports before claiming
reproduction. Generation supports `--resume` with an unchanged source snapshot,
input plan, precision, device map, and settings. Each model suite also supports
`--resume` with matching input, source, environment, and sampling settings.

### Independent service and disconnect recovery

On 10 September, `ijcv-complexity.service` was installed as an enabled systemd
user service. Its unit file is at
`/home/riccardo/.config/systemd/user/ijcv-complexity.service`, and its implementation
is `ijcv_persistent_supervisor.py` in the artifact directory. User lingering was
enabled with `loginctl enable-linger riccardo`, so the user service manager remains
available after the last SSH or VS Code connection closes.

The service replaced only the idle original continuation dispatcher. Existing
generation and fitting jobs were left running to preserve in-flight computation.
It observes their processes until they exit. If an original process is terminated
with its launch backend, the service waits until the associated writers are gone
and resumes the incomplete stage from its recorded checkpoints. Future stages
and resumed jobs run directly under systemd, independently of VS Code and Codex.
The scientific source snapshot, datasets, settings, and completed posteriors are
unchanged. Recorded scientific failures remain failures requiring review.

Completed generation inputs retain their saved random states; an interrupted
image may be recomputed. Completed model fits are reused. In-progress Bayesian
sampling and CV do not have intermediate checkpoints, so an unfinished fit or
CV stage may repeat if its original process is terminated. Connection recovery
preserves completed checkpoints, not every unsaved sampler iteration. Interrupted
posterior writes without completed diagnostics are archived for inspection.

Monitor the service and its journal with:

```bash
systemctl --user status ijcv-complexity.service
journalctl --user -u ijcv-complexity.service -f
```

The earlier image-progress and model-log commands still work. Resumed job logs
are written under `supervisor_logs/`. `supervisor_status.json` distinguishes
observing an existing job, waiting for dependencies, running under systemd,
completion, and failure. The three initial continuation status files are
superseded by this report. Do not run the original dispatchers alongside the
service, as that could launch duplicate work.

A controlled recovery test terminated a disposable original job after it saved
one checkpoint. An independent test service automatically resumed it, preserved
the completed checkpoint, and ran the resumed process in the service's cgroup.
Evidence is in `service_recovery_test/verification.json`. The real fitting and
generation jobs were not interrupted during this test.

The service exits after the requested branches finish, and writes
`full_run_status.json`. Inspect numerical verification reports separately from
service exit status. After the experiment, automatic startup can be removed with
`systemctl --user disable ijcv-complexity.service`. Lingering remains enabled;
it can be restored to its previous setting with `loginctl disable-linger riccardo`
once it is no longer needed by any unattended user services.

## Numerical comparison policy

### Completed M2 checks from saved COCO scanpaths

All four full M2 fits have completed. The exported rankings were compared with
their historical CSVs before waiting for the remaining candidate models:

| Ranking | Rows | Byte-identical | Spearman | Score RMSE |
| --- | --- | --- | --- | --- |
| Human count | 2,241 | Yes | 1.0 | 0 |
| Generated count | 2,086 | Yes | 1.0 | 0 |
| Generated RT | 2,086 | Yes | 1.0 | 0 |
| Human RT | 2,241 | No | 0.9999847591 | 0.0011943168 |

The human RT maximum absolute score difference is 0.0067894690. All four fits
have zero divergences, maximum image-effect R-hat below 1.005, minimum bulk ESS
above 3,700, and minimum tail ESS above 1,500. Detailed evidence is retained in
`early_m2/comparison.json` and each suite's diagnostics JSON. These checks use
saved scanpaths; they do not establish reproduction from regenerated images or
agreement of the full LOO/CV tables. Those tables are now complete, with the
differences and scoring correction documented in the investigation.

### Published tables

[paper_tables.csv](../complexity/reference/paper_tables.csv) transcribes 173
reported cells from the supplied paper's Tables 1, 2, 3, and A1. It preserves
small inconsistencies between Tables 1 and A1 instead of silently reconciling
them. The comparison reports observed values, signed differences, and whether
they agree at the printed precision:

```bash
"$RUN_ROOT/venvs/measurement/bin/python" -m complexity.evaluation.compare_paper_tables \
  --run "$RUN_ROOT/historical" --output "$RUN_ROOT/historical/paper_comparison"
```

That archived root retains the initial joint RT LOO scores. The completed
14 September rescoring preserves them and supplies corrected scores separately.
For the current comparison, use the small overlay containing corrected LOO and
the original CV/count tables:

```bash
"$RUN_ROOT/venvs/measurement/bin/python" -m complexity.evaluation.compare_paper_tables \
  --run "$RUN_ROOT/investigation_2026-09-14/corrected_historical" \
  --output /path/to/new/table-comparison
```

All 173 cells remain available; 54 match the printed values and 119 differ.
The [investigation](complexity_discrepancies.md) distinguishes the two small
human-count differences from the 117 RT table differences and records all
27 completed RT LOO evaluations, including the 15 corrected joint evaluations.

Missing and non-finite results have explicit statuses. Exact agreement at printed
precision is a descriptive check, not an assumed guarantee for Monte Carlo fits.
Rank comparisons retain the existing Figure 4a reference choice, human M2-N,
matching the author's pending caption correction. The NSD verification retains
the previously documented label identity, ranking, and sampling criteria.

Seven scientific regression tests pass, including the partition schedule,
restored joint marginal LOO scoring, explicit changed-label reporting, and
reporting of missing, non-finite, and numerically different table results. No
code-style enforcement tools were installed or added.
