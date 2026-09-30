# Behavioural complexity estimation

## Purpose and paper mapping

This component estimates visual-search difficulty from response times and eye movements. COCO workflows
support Tables 1-3 and A1 and ranking comparisons in Figures 3-4. The NSD M2 count workflow supplies the
image-target labels used by Routes A and B and the label-side variance analysis.

## Workflow

| Profile | Population and operations |
| --- | --- |
| `coco` | Human and successful synthetic target-present trials; RT/count suites, Mean-N baseline and figure comparisons. |
| `coco-rt-tables` | Unfiltered synthetic target-present trials; RT suite, ranking export and Tables 1/A1 comparison. |
| `nsd` | Successful synthetic trials, NSD target correction, positive-count M2 fit, ranking export and variance. |
| `all` | The `coco` and `nsd` workflows. |

Stages are `generate`, `prepare`, `fit`, `export`, `compare`, `variance` and `verify`.
Generation is explicitly selected. The RT-table profile defaults to `prepare fit export compare` and does
not accept `variance` or `verify`. Other profiles default to `prepare fit export compare variance verify`;
COCO comparisons and NSD variance/verification run where applicable.

`RT` is in milliseconds. The standard conversion defines `N` as fixation count minus one, so it counts
saccades. Count models and Mean-N keep `N > 0`. Human COCO selection retains target-present trials for
filenames with at least one observer fixation inside the target box. Synthetic figure trials additionally
require `correct == 1`. The RT-table profile does not filter on that flag.

Trial counters follow each subject's retained JSON order. Image effects use filename identity. Ranking
export selects the last encountered target for a filename and sorts posterior means in descending order.
The exported score is `img_re` or, for joint models, `C_image`; larger values indicate greater difficulty.
Mean-N exports raw average counts and its first encountered task mapping.

NSD correction applies the metadata crop transform at 425 by 425 pixels, uses the first usable matching
category box, updates target hits, removes knife, and rejects invalid or oversized boxes (area above 90%).

## Inputs, identities and sources

Obtain the underlying [COCO-Search18](https://sites.google.com/view/cocosearch/),
[COCO](https://cocodataset.org/#download) and [NSD](https://www.naturalscenesdataset.org/) data externally.
Commands start from the prepared inputs below; the example configuration names each path explicitly.

| Input | Required schema or layout |
| --- | --- |
| Scanpath JSON, optionally gzip for direct conversion | List of records with `name`, `subject` or `sample_id`, `task`, `condition`, `X`, `Y`, `T`, `RT`, `length`, `bbox`, `correct`; optional `split` and `fixOnTarget`. |
| Prepared trial CSV | `RT,subject,image,trial,N,T2T,TTFix2R` plus target, condition, correctness and bounding-box fields. |
| NSD metadata JSON | `images` with path, original size, fractional crop box and category-instance boxes; `categories.instances` maps names to IDs. |
| Generation image directory | Images below target-category subdirectories. |
| Generation bounding-box JSON | Image-name keys mapped to objects with `bbox: [x,y,width,height]` in displayed-image coordinates. |
| Existing posterior | NetCDF with posterior image coordinates and `img_re` or `C_image`, matching the trial table. |

The fixed [nsd_m2_ranking.csv](nsd_m2_ranking.csv) has columns `image,score,task` and 12,447 ranking rows.
These represent 10,291 physical NSD images and 11,641 physical image-target pairs. Training selection
retains 12,108 records, 9,990 physical images and 11,304 physical image-target pairs. Do not rewrite labels,
reorder scores or deduplicate the file when using it as supervision.

Its SHA-256 is `2101a2143b00da45625afeaeda08880c70761f825efb0a7c17769624c0a0e6fb`.
Installed code resolves this resource inside the `complexity` package.

## Installation

Use Python 3.12 and a C++ compiler for the normal PyTensor backend. From the repository root:

```bash
python3.12 -m venv .venv-complexity
source .venv-complexity/bin/activate
python -m pip install -c complexity/requirements.lock.txt '.[complexity]'
```

[requirements.lock.txt](requirements.lock.txt) records the measurement environment, including PyMC 5.27.0,
ArviZ 0.23.0, NumPy 2.4.2 and pandas 3.0.1. ScanDiff uses the separate environment described below.

## Commands

Copy [config.example.json](config.example.json), edit the relevant inputs and select an external output
directory. JSON-relative paths resolve beside the configuration file after environment-variable and `~`
expansion. Only inputs used by the selected profile and stages are required.

```bash
export COMPLEXITY_DATA=/path/to/prepared/scanpaths
export BRAIN_DATA=/path/to/prepared/images-and-metadata
export COMPLEXITY_OUTPUT=/path/to/output/nsd
cp complexity/config.example.json /path/to/nsd.json
python -m complexity run --config /path/to/nsd.json --profile nsd
```

Replay deterministic preparation and export from an existing posterior:

```bash
python -m complexity run --config /path/to/nsd.json --profile nsd \
  --stages prepare export variance verify --posterior-file /path/to/M2_n_studentT.nc
```

Use distinct output directories in the configurations for the figure and RT-table workflows:

```bash
python -m complexity run --config /path/to/coco-figures.json --profile coco
python -m complexity run --config /path/to/coco-rt-tables.json --profile coco-rt-tables
```

`coco_models.rt` and `coco_models.count` may restrict the selected model names; omit them for full suites.
The model modules' `--help` lists those names. Add `--resume` to reuse compatible fit or generation state.
Source, inputs, environment and settings must still match. Export-only `--posterior-file` is for NSD and
cannot be combined with the `fit` stage.

Retained utility modules also accept explicit paths:

```bash
python -m complexity.evaluation.compare_paper_tables \
  --results-dir /path/to/coco-rt-output --tables 1 A1 --output-dir /path/to/table-comparison
python -m complexity.evaluation.compute_variance --ranking-file complexity/nsd_m2_ranking.csv --min-pairs 2
python -m complexity.evaluation.rescore_rt_loo --results-dir /path/to/saved-rt-fit \
  --trials-file /path/to/trials.csv --output-dir /path/to/new-loo-results
python -m complexity.evaluation.compare_ranking_pair --left-file /path/to/left.csv \
  --right-file /path/to/right.csv --output-dir /path/to/ranking-comparison
```

### Regenerate scanpaths from prepared images

Supply an external ScanDiff checkout containing `configs/`, `src/`, its visual-search checkpoint and task
embeddings. The adapter imports the external model and diffusion implementation. It uses
`timm` model `vit_base_patch14_reg4_dinov2.lvd142m`, ten simulated viewers per image, and seed 1000.
COCO configuration uses float32 on one device; the NSD example uses AMP and deterministic image partitions
across four devices, with worker seed `1000 + worker_index`.

Create a separate environment from the repository root:

```bash
python3.12 -m venv .venv-scanpaths
.venv-scanpaths/bin/python -m pip install -c complexity/generation/requirements.lock.txt '.[scanpaths]'
export SCANDIFF_ROOT=/path/to/ScanDiff
export SCANDIFF_PYTHON=/absolute/path/to/.venv-scanpaths/bin/python
python -m complexity run --config /path/to/nsd.json --profile nsd --stages generate prepare
```

The `generation.scandiff_root` and `generation.python_file` fields select that source tree and interpreter.
The generation environment retains `timm==1.0.9` in its
[lockfile](generation/requirements.lock.txt). Prepared NSD boxes and metadata must describe the same crop.
For direct invocation, use `python -m complexity.generation.generate_scanpaths --help` in that environment;
`--checkpoint-file` and `--task-embeddings-file` can override the default artifact paths.

| Required artifact below ScanDiff root | SHA-256 |
| --- | --- |
| `checkpoints/scandiff_visualsearch.pth` | `8a1683c46dcf6ed3b0e08ca6128bacd6354a1bd8f7824e6fb203a765e43201a6` |
| `data/task_embeddings.npy` | `831ca726ff8c192cd1cca37140f0c746d5acb61e88aa8b9f838bf26240e8281f` |
| `configs/demo.yaml` | `82c1545aaf7a137ee4b61c884cfdb27c8fd83e1d0f92381ddf08f27c7e90cf6e` |

Model weights and extracted features remain external. `TORCH_HOME` and `HF_HOME` select the libraries'
weight caches. Generation writes settings, RNG recovery states, per-image progress and combined JSONs to
its configured output; keep these when resuming an interrupted generation.

## Outputs

| Location below `output_dir` | Contents |
| --- | --- |
| `trials/` | Human, successful synthetic and corrected NSD trial CSVs; RT-table input is `coco_unfiltered.csv`. |
| `coco_human_rt`, `coco_human_count` | Human model posteriors, settings, diagnostics, LOO and CV results. |
| `coco_predicted_rt`, `coco_predicted_count` | Successful synthetic figure-model results. |
| `coco_unfiltered_rt/` | Separate unfiltered RT-table model results. |
| `nsd_fit/` | NSD M2 posterior and diagnostics. |
| `rankings/` | `image,score,task` CSVs and distributions, including `coco_unfiltered_rt/` and `predicted_mean/`. |
| `comparisons/` | Figure ranking comparisons or RT-table cell comparisons. |
| `verification.json`, `last_run.json`, `logs/` | Ranking check, invocation settings and stage logs. |

## Paper targets and reproducibility settings

The standard frozen inputs select 24,880 human RT trials (2,241 filename labels), 17,310 successful
synthetic RT trials (2,086 labels), and 31,010 unfiltered synthetic RT trials (2,779 labels).
Positive-count filtering leaves 24,643 human and 17,309 successful synthetic observations.
NSD correction yields 83,508 trials, of which 82,221 have positive counts.

Full fits use four chains, 2,000 tuning steps, 2,000 retained draws, target acceptance 0.95 and seed 42.
Cell-held-out CV uses five folds, two chains, 600 tuning steps, 600 draws and seed 42; joint CV uses two
Monte Carlo count draws. Joint LOO uses the marginal RT likelihood with at least 20 count draws.
The model files retain family-specific priors, likelihoods and historical CV parameter expansion.

[paper_tables.csv](paper_tables.csv) transcribes the printed cells of Tables 1, 2, 3 and A1. Its columns are
`table,suite,file,model,metric,published,decimals`; the suite/file fields identify the expected output and
`decimals` specifies printed precision. Tables 1/A1 map to `coco_unfiltered_rt`; count tables map to their
human/synthetic suites. The checker reports matching, different, missing and non-finite cells without
changing reference values.

The saved-posterior export check requires identical labels and order and maximum score difference
`1e-12`. Fresh NSD fitting uses identical label membership, Spearman at least `0.99`, score RMSE at most
`0.02`, image-effect R-hat at most `1.01`, bulk/tail ESS at least `400`, and zero divergences. These criteria
apply to that workflow. Sampling diagnostics and PSIS-LOO reliability are separate quantities.
