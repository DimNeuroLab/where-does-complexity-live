# Complexity code inventory

Inventory date: 2026-09-09.

This inventory maps the supplied IJCV manuscript to the existing
`projects/complexity/` research workspace. Paths below are relative to that
workspace unless a different project is named. They describe the original source
material. The subsequent copy stage is documented in the
[complexity README](../complexity/README.md), including the filename mapping.

The reference manuscript is `/home/riccardo/AIPA_IJCV_26.pdf`, the 50-page version
also used for the README figure. The smaller `AIPA_IJCV_26.pdf` inside the
complexity workspace and `method_section.tex` are historical material, not the
reference for publication scope.

## Scope and evidence

The complexity component covers:

1. Scanpath conversion and behavioural-data preparation.
2. RT-based measurement models and selection of M2.
3. Count-based models, the Mean-N baseline, and ranking comparisons.
4. Complexity-label generation from ScanDiff scanpaths on NSD.
5. Ranking exports and qualitative examples.

The relevant manuscript material is §4.1, §4.1.1, §5.3, Tables 1-3, Figures 3-4,
and Appendix A. The image and fMRI predictors belong to Routes A and B.
The variance analysis in §5.3 has related code in `preds/analysis.py`, outside
this workspace, and should ultimately be coordinated with cross-route analysis.

Evidence levels used here:

- **Verified:** file identity or a numerical result was checked directly.
- **Candidate:** code implements the required operation, but the exact historical
  run or final figure has not been established.
- **Unresolved:** local artifacts differ from the paper or required provenance
  is unavailable.

During this inventory, no model fitting, code cleanup, data modification, or
migration was performed.
Checks used source inspection, CSV summaries, numerical comparisons, manifests,
and figure-image comparisons. Posterior sampling and convergence were not rerun.

## Map from the paper to implementation

| Paper component | Source implementation | Existing evidence and status |
| --- | --- | --- |
| §4.1: trial-level RT and movement counts | `conv.py` | **Candidate.** Converts scanpath JSON to CSV, derives saccades or fixations, assigns trial indices, and optionally filters individual rows. The paper's exact inclusion procedure needs confirmation. |
| Table 1: RT model comparison | `estimate_complexity_gpt_final_2.py` | **Candidate.** Contains M1-M4, all five joint variants, marginal RT likelihoods, Jacobian correction, PSIS-LOO, and cell-held-out CV. Original GT fit and LOO directories referenced in the notes are absent here. |
| Table A1: RT CV and ranking stability | Same RT script; `model_selection_table.csv` | **Verified numerical correspondence.** The CSV contains the nine model rows and reported rounded metrics. It is a summary, not the original fold outputs. |
| Table 2: human count-model comparison | `estimate_complexity_fixations.py` | **Candidate.** Contains M1-M4 and the corresponding LOO/CV machinery. `difficulty_rankings_gt_fixations/` contains rankings for all four models, but original GT posterior and LOO files are absent here. |
| Table 3: ScanDiff count-model comparison | Same count script | **Candidate.** `difficulty_rankings_scandiff_fixations/` contains all four model rankings. The original fit, LOO, and CV directory is absent here. |
| Figure 3a: GT M2-N versus GT M2-RT | `compare_two_rankings.py` | **Verified numerical correspondence.** The current rankings reproduce Spearman 0.912774 and top-20 overlap 0.80, consistent with the stored summary. They overlap on 2,241 image names. |
| Figure 3b: Mean-N versus GT M2-RT | `compute_ranking_mean_n.py`, `compare_two_rankings.py` | **Verified and selected by the author.** The generated-count baseline from `merged_correct.csv` reproduces the saved ranking byte-for-byte, Spearman 0.701049 and overlap 0.30. Restored to the publication pipeline on 10 September. |
| Figure 4a: ScanDiff M2-N versus the stated GT M2-RT reference | `compare_rankings.py`, `compare_two_rankings.py` | **Unresolved.** The reported 0.711098 correlation and 0.35 overlap are reproduced using GT M2-N as the reference. Using GT M2-RT gives 0.655243 and 0.20 with the current files. |
| Figure 4b: ScanDiff M2-RT versus GT M2-RT | `compare_two_rankings.py` | **Verified numerical correspondence.** `difficulty_rankings_scandiff_correct/` reproduces Spearman 0.529245 and overlap 0.10 against GT M2-RT on 2,086 image names. |
| §4.1.1 and §5.3: NSD M2 complexity labels | `conv.py`, `correct_nsd_errors.py`, `estimate_complexity_fixations.py`, `visualize_difficulty_distribution.py` | **Strong artifact linkage.** The corrected CSV's positive-N rows match the `results_nsd_no_knife/` manifest counts. Its exported M2 ranking is byte-identical to the ranking in `preds`. Exact run arguments still need to be recorded. |
| Figure A1: GT RT ranking examples | `visualize_difficulty.py` | **Candidate.** `difficulty_rankings_gt/ranking_M2_logrt_studentT.png` exists, but an exact pixel match to the paper figure was not found. |
| Figure A2: NSD ranking examples | `visualize_difficulty.py` | **Verified figure identity.** `difficulty_rankings_nsd_no_knife/ranking_M2_n_studentT.png` has identical decoded RGB pixels to the image embedded in the paper. |
| §5.3: target-driven variance | `preds/scripts/12_compute_complexity_variance.py` | **Verified.** Re-running the script on the Route B reference exactly reproduces the author's output: total 0.072235, within 0.034069 (47.16%), between 0.038166 (52.84%), on 1,205 images. Repeated physical scene-target records are averaged first. |
| Route B prediction variance | `preds/scripts/13_compute_predicted_complexity_variance.py` | **Verified against the author's supplied output.** CPU inference from the existing fold checkpoints gives total 0.025626, within 0.007898 (30.82%), between 0.017728 (69.18%), on 1,173 images. |

The exact raster assets for Figures 3, 4, and A1 were not found by pixel identity.
Their candidate plotting code and numerical evidence are separate from that
asset-level check; differences could include cropping or later figure editing.

## Inventory of all 19 Python files

### Core migration candidates

These implement operations needed by the paper, even where historical inputs or
outputs still need to be reconciled.

| File | Role | Migration note |
| --- | --- | --- |
| `conv.py` | JSON-to-CSV conversion; RT, N, target-hit timing, trial indices | Make input conventions, units, target keys, and filtering explicit. It reads RT from the JSON; it does not generate ScanDiff trajectories or synthesize a missing RT field. |
| `estimate_complexity_gpt_final_2.py` | RT and joint model definitions, posterior fitting, LOO, CV, stability | Retain the joint variants because they appear in Tables 1 and A1. Only M1 and M2 are currently enabled in the model registries. |
| `estimate_complexity_fixations.py` | Count models and NSD label fitting | Retain M1-M4 for Tables 2-3 and M2 for label production. Only M1 and M2 are currently enabled. |
| `compute_ranking_mean_n.py` | Mean-N baseline and its CV stability | Historical input is `merged_correct.csv`. The publication pipeline explicitly uses the equivalent successful generated COCO table. |
| `compare_two_rankings.py` | Explicit pairwise ranking comparisons and scatter plots | Best candidate for the four named comparisons in Figures 3-4. |
| `compare_rankings.py` | Batch comparison of matching ranking filenames | Shares comparison logic with the pairwise script. Its same-filename matching naturally compares the same model family on both datasets. |
| `visualize_difficulty_distribution.py` | Posterior-mean CSV export and distributions | Contains a necessary label-export operation despite its plotting-oriented name. Currently exports only `image`, `score`, and `task`, without posterior SD. |
| `visualize_difficulty.py` | Hardest, median, and easiest image montages | Supports Appendix A. Current paths target the corrected NSD run. |

### Conditional preprocessing dependency

| File | Role | Evidence |
| --- | --- | --- |
| `correct_nsd_errors.py` | Recompute NSD target bounding boxes and correctness; filter trials | Likely part of the final label lineage. Transforms boxes into 425-by-425 coordinates, keeps correct trials, removes `knife`, and rejects invalid boxes or boxes exceeding 90% of image area. Those exact choices need to be documented and confirmed as publication preprocessing. |

### Supporting utilities without a direct published result established

| File | Actual role | Proposed disposition |
| --- | --- | --- |
| `compare_rt_correlations.py` | Compare per-image mean RT and fixation count between two scanpath datasets | Optional diagnostic. It is not a direct RT-versus-N correlation analysis within one dataset. |
| `visualize_scanpaths_by_difficulty.py` | Inspect scanpaths and target boxes on ranked images | Optional diagnostic useful for preprocessing review. No direct paper figure match established. |
| `visualize_test_scanpaths.py` | Inspect example scanpaths | Optional diagnostic, not a scientific test suite. |

### Outside the demonstrated publication scope

These should stay in the research workspace unless a paper dependency is shown.
This recommendation is not a request to delete them.

| File | Experiment or purpose |
| --- | --- |
| `compare_ic9600_rankings.py` | Compare predicted difficulty against IC9600 ratings. |
| `export_ic9600_comparisons.py` | Export IC9600 comparison data. |
| `compare_savoias_rankings.py` | Compare against SAVOIAS ratings. |
| `count_class_imgs.py` | COCO category counts and image selection. |
| `count_ppl_imgs.py` | Person-category filtering and counts. |
| `create_filtered_csv.py` | Flag NSD images containing people or animals. |
| `filter_images_by_bbox.py` | Alternative NSD exclusion of targets touching image boundaries. |

The manuscript mentions IC9600 and SAVOIAS in related work, but does not report
these workspace experiments. Gazeformer comparisons, NSD free-viewing label runs,
and the `nsd_no_cropped` alternative likewise have no established result in the
supplied paper. Directory names containing `test` are not evidence of automated tests.

## Data and artifact lineage

### Trial tables

Counts below describe the files as they exist, before any new preprocessing.
Every row in these tables uses `N = n_fixations - 1`.

| CSV | Rows | Unique image strings | Tasks | Rows with N = 0 |
| --- | ---: | ---: | ---: | ---: |
| `merged_gt.csv` | 24,880 | 2,241 | 18 | 237 |
| `merged_correct.csv` | 17,310 | 2,086 | 18 | 1 |
| `nsd_merged.csv` | 116,848 | 17,173 | 17 | 1,844 |
| `nsd_merged_corrected.csv` | 83,508 | 12,449 | 16 | 1,287 |
| `filtered_nsd_merged.csv` | 56,209 | 9,276 | 17 | 322 |

The count model discards N <= 0. For the corrected NSD table this leaves 82,221
rows and 12,447 image strings, exactly the counts recorded in
`results_nsd_no_knife/manifest.json`.

`merged_gt.csv` starts with records matching the human
`coco_search18_fixations_TP_merged.json`. `merged_correct.csv` starts with records
matching `coco_search18_all_scanpaths.json`, the candidate ScanDiff source.
These are record-level observations, not a substitute for generation logs.
The latter file should not be labeled human GT merely because its dimensions
match the manuscript's stated 17,310 trials and 2,086 images.

### Final NSD label candidates

The strongest confirmed downstream link is:

```text
nsd_merged.csv + nsd_all_scanpaths.json + external NSD metadata
  -> correct_nsd_errors.py
  -> nsd_merged_corrected.csv
  -> positive-N filtering and M2 fitting
  -> results_nsd_no_knife/M2_n_studentT.nc
  -> posterior-mean ranking export
  -> difficulty_rankings_nsd_no_knife/full_ranking_M2_n_studentT.csv
  -> identical file in projects/preds/
```

The conversion and fitting arrows are supported by code and matching manifest
counts, but the historical command lines have not been recovered. The final CSV
identity is verified by SHA-256:
`2101a2143b00da45625afeaeda08880c70761f825efb0a7c17769624c0a0e6fb`.

Other NSD M2 rankings have 17,172 rows (`nsd_fixations`), 9,275 rows
(`nsd_no_cropped`), and 17,735 rows (`nsd_freeview_fixations`). None is byte-identical
to the ranking in `preds`.

### Saved outputs and dependencies

- `difficulty_rankings_gt/`, `difficulty_rankings_gt_fixations/`,
  `difficulty_rankings_scandiff_correct/`, `difficulty_rankings_scandiff_fixations/`,
  and `difficulty_rankings_baseline/` contain candidate paper ranking artifacts.
- `comparison_results_rt/`, `comparison_results_baseline/`,
  `ranking_comparisons_correct/`, and `ranking_comparisons_fixations/` contain
  relevant comparison summaries or plots. Some `.csv` summaries are actually
  whitespace-formatted text; preserve their provenance before normalizing them.
- `model_selection_table.csv` is the available RT CV summary for Appendix A.
  `notes.md` records additional historical metrics but disagrees with the current
  paper on some values, including GT M2-RT ranking stability.
- The notes reference `results_gem/`, `results_gem_fixations_gt/`, and
  `results_gem_fixations/`; these directories are not present in the workspace.
  Consequently, Tables 1-3 cannot yet be independently verified from their
  original posterior and LOO/CV artifacts here.
- Six posterior `.nc` files remain under three NSD results directories. Each is
  approximately 6.5-9.1 GB. These belong in external research-artifact storage,
  not in a source-code migration.
- Image files and NSD metadata referenced under
  `/data/COMMON/brain_activation/` exist on this machine. They are external
  dependencies, not portable repository paths. The metadata file is about 2.5 GB.
- No ScanDiff inference implementation, checkpoint selection, or exact generation
  command is present among the 19 Python files. The provided trajectory JSONs
  need a documented upstream source and generation recipe.
- The core scripts import NumPy, pandas, PyMC, ArviZ, xarray, PyTensor, SciPy,
  Matplotlib, seaborn, and Pillow. Exact historical versions and the NetCDF
  backend still need to be established. `pycocotools` appears in the auxiliary
  category-counting utilities.

## Follow-up: shared-data investigation

The follow-up inspected `/data/COMMON/brain_activation/`, including the available
scripts, notebooks, configurations, logs, and artifact filenames. The relevant
prepared images, bounding boxes, and NSD metadata were found. No ScanDiff
inference implementation or checkpoint was identified, and no complexity
posterior `.nc` files were found there. The directory named `scandiff` contains
SAVOIAS and IC9600 data, not the missing inference pipeline.

The following points are now settled for reproducing the existing artifacts:

- The shared `coco_search_18/coco_search18_fixations_TP_merged.json` is byte-identical
  to the human JSON in the research project.
- The shared NSD image tree contains all image-target paths referenced by
  `nsd_merged.csv`. It has 52,486 prepared images across 17 target directories.
  The missing `bowl` directory is consistent with its absence before NSD correction;
  `knife` is removed later by the correction script.
- The complete metadata file is available at
  `nsd_augmentation_data/metadata.json`. `metadata_small.json` contains only 159
  image records and is not a replacement for the full metadata.
- Running the unchanged converter with `--movement saccades --only_condition
  present --only_correct 1` exactly reproduces both `merged_correct.csv` from
  `coco_search18_all_scanpaths.json` and `nsd_merged.csv` from
  `nsd_all_scanpaths.json`, including their CSV bytes.
- Running the unchanged NSD correction script with that shared metadata exactly
  reproduces all 83,508 rows and bytes of `nsd_merged_corrected.csv`. The run
  corrects 106,919 bounding boxes and changes 21,100 correctness flags, then keeps
  95,748 successful trials, 91,069 after removing `knife`, and 83,508 after the
  bounding-box validity filters.
- Positive-N filtering yields observations, subject indices, image indices, and
  centered trial covariates that exactly match the saved M2 posterior's input
  datasets. Its image-coordinate order also matches the current preparation code.
- The final NSD M2 posterior records PyMC 5.27.0, ArviZ 0.23.0, four chains,
  2,000 retained draws per chain, and 2,000 tuning steps. These are verified
  artifact metadata, not inferred defaults. Seeds and target acceptance were not
  recovered from those attributes.
- Reading the saved M2 posterior directly reproduces the reference ranking order
  and scores to a maximum absolute difference of 1.11e-16. This verifies the
  posterior-to-ranking link independently of the export script. Running the
  unchanged exporter was blocked by absent ArviZ in the available `preds`
  environment; that environment also lacks PyMC, PyTensor, xarray, h5netcdf, and
  seaborn.

All regenerated CSVs were written in temporary directories and removed after
comparison. No research source or stored input/output artifact was changed, and
no Bayesian fitting was performed.

## Follow-up: recovered ScanDiff source and variance analysis

Read access is now available to
`/home/valentyn/data/Complexity/ScanDiff/` (resolved path
`/data/valentyn/Complexity/ScanDiff/`). It contains `generate_scanpaths.py`,
Hydra configuration, `checkpoints/scandiff_visualsearch.pth`, task embeddings,
and the saved COCO and NSD generation outputs.

- `results_cocosearch18/all_scanpaths.json` is byte-identical to the research
  workspace's `coco_search18_all_scanpaths.json`: 31,010 scanpaths across 3,101
  image-target filenames.
- Every one of the 52,481 JSON files under `results_nsd/<task>/` matches the
  corresponding records in `nsd_all_scanpaths.json`, covering all 524,810
  scanpaths, with no missing or different records. This comparison checks
  parsed records, not concatenated JSON bytes or aggregate ordering.
- The generator selects the visual-search checkpoint, ten simulated viewers,
  and worker seeds `1000 + process_id`. It sorts input paths and partitions them
  across available GPUs. On 10 September the author confirmed four GPUs and
  recalled a command containing `--force`. With that flag, existing outputs are
  regenerated. The recovered partition rule is contiguous sorted chunks of size
  `ceil(N / 4)`, with worker seeds 1000 through 1003.
- The current generator calls `save_image_scanpaths(PIL_image, ...)`, but
  `PIL_image` is undefined. This is an execution defect to fix in the migrated
  copy; finding this source does not mean its current bytes can generate the
  historical outputs unchanged. Some lines in `requirements.txt` also combine
  several package specifications and need normalization before installation.

Both variance scripts in `preds/scripts/` were run successfully. The first uses
the existing reference CSV; the second uses the saved Route B fold checkpoints
and CPU inference, with no model training. Both reproduce every printed value
in the author's supplied outputs. The previous 48.30% and 48.47% calculations
were incomplete checks: they counted distinct targets to select scenes but
retained repeated records when computing variance. The verified scripts first
average each physical scene-target pair. The 1,205 versus 1,173 scene counts
reflect the full label set versus the subset usable for fMRI predictions.

## Migration decisions and initial rerun status

This list covers both executing the complete image-to-ranking chain and matching
all complexity results in the manuscript. Paper-consistency questions do not
prevent an execution-only replay of the existing NSD behavior.

The author has clarified that the disabled models were intentional for later
runs and may be re-enabled. Changes to paths and other execution conveniences
are also authorized. Items 2, 5, and 9 include implementation work we can handle;
they do not all require recovery of historical files or an author decision.

The author subsequently chose to restore the historical generated-count Mean-N
baseline, retain the existing COCO filename grouping for this release, and use
the Route B ranking as the reference. The execution implementation and actual
validation outcomes are now documented in
[`complexity_validation.md`](complexity_validation.md) and the
[component README](../complexity/README.md). The numbered findings below preserve
the provenance investigation; setup tasks in that list are not new approval gates.

1. **ScanDiff generation recipe: source and artifacts recovered.** The recovered
   generator and saved outputs establish the upstream location. Prepare a working
   environment, fix execution defects, and record an explicit generation recipe.
   Four GPUs are now confirmed, and `--force` was recalled by the author. The
   current adapter still needs the original four-worker partitioning for a
   matching full generation run. Prepared images and bounding boxes are available.

2. **Fitting environment and remaining sampling settings.** Build a reproducible
   environment around the confirmed PyMC 5.27.0 and ArviZ 0.23.0 versions, including
   compatible Python, PyTensor, NumPy, and NetCDF dependencies. Record the
   new reproduction settings. The author accepted using seed 42 and target
   acceptance 0.95 without claiming they were recovered from the historical run.
   Four chains, 2,000 retained draws, and 2,000 tuning steps are already confirmed
   for final NSD M2.

3. **COCO input selection and success filtering.** Establish the authoritative
   input for each human, predicted, and Mean-N experiment. The baseline exactly
   matches `merged_correct.csv`, not means computed from `merged_gt.csv`. The
   manuscript's 17,310 trials and 2,086 images match the former; GT rankings
   contain 2,241 image names. Reconcile the manuscript's at-least-one-successful-
   observer inclusion rule with the now-verified per-trial success filters.
   Reproducing those filters is settled; whether they implement the intended
   scientific protocol is not.

   **Decision, 10 September:** restore Mean-N to the successful generated COCO
   table, matching the historical baseline. The pipeline now uses
   `trials/coco_predicted.csv` and Figure 3b reads `rankings/predicted_mean/`.
   The earlier human-baseline experiment below is retained as investigation history.

   Additional checks: successful human trials alone give 22,946 rows and 2,230
   images. Keeping all human trials for images with at least one successful
   observer gives 24,770 rows and 2,230 images. Neither reproduces the stated
   17,310 rows and 2,086 images. For positive-N model inputs, the full human table
   gives 24,643 rows, while the successful predicted table gives 17,309 rows.

   The substantive baseline question is whether Figure 3b is intended to compare
   estimators on the same human measurements. Its saved baseline uses generated
   measurements, changing both the estimator and the data source. Direct mean
   positive-N scores from the existing human table have Spearman 0.896439 against
   human M2-RT across 2,241 filenames. The generated Mean-N baseline has 0.701049
   across 2,086 filenames. These illustrative checks use each table's own
   available image set; a controlled comparison should also align the subsets.

4. **COCO image-target identity.** Decide whether effects and ranking joins should
   use image alone or image-target pairs. Existing code uses image alone; 233 GT
   image names and 75 predicted-image names occur under multiple tasks. The
   shared COCO bounding-box file is also keyed by image alone. Changing these
   keys would change the scientific experiment and must not be treated as a
   cosmetic cleanup.

   Example: human image `000000009527.jpg` has ten positive-N trials searching
   for `bottle` (mean N 4.3) and ten searching for `bowl` (mean N 3.8). Both sets
   currently contribute to one image effect. There are 2,489 distinct human
   image-target pairs across 2,241 filenames, and 2,163 predicted pairs across
   2,086 filenames. NSD target suffixes already distinguish different tasks in
   the current ranking filenames, so this particular target collision is COCO-specific.

   Executing `prepare_df` from both original fitting scripts assigns index 34 to
   both targets in this example. The two task-directory image files have identical
   SHA-256 hashes. Both saved human M2 rankings contain only one row for this
   filename, labeled `bowl`, confirming that the collision reaches the exports.
   No filename in `nsd_merged_corrected.csv` occurs under multiple tasks. This
   establishes the specific COCO collision, not a blanket validation of NSD.

5. **Complete model comparison runs.** Specify the model registry and CV settings
   needed for Tables 1-3 and A1. Both current runners enable only M1 and M2;
   M3, M4, and the joint variants are implemented but disabled. Recover original
   COCO fit/LOO/CV artifacts where possible; none were found in the shared
   directory. The runners' CV defaults are 600 draws, 600 tuning steps, and two
   chains, but the historical settings for each paper comparison are unverified.

6. **Figure 4a reference.** Resolve whether the reference should be human M2-N or
   human M2-RT. Current artifacts reproduce the reported Spearman 0.711098 and
   top-20 overlap 0.35 against M2-N. Against M2-RT, they give 0.655243 and 0.20.
   The paper currently describes M2-RT. The author will correct and update the paper.

7. **NSD variance result: resolved.** The verified ground-truth analysis first
   averages repeated physical scene-target records and gives 47.16%, rounding
   to the paper's 47.2%, on 1,205 images. The author's prediction output is also
   reproduced exactly, with 30.82% on 1,173 images. Use the two verified scripts
   identified above for the respective analyses.

8. **Shared output for Routes A and B: reference selected.** The author selected
   the 12,447-row ranking used by Route B as the reference. The paper's Route A
   feature comparisons report 12,476 samples; tracing that older count remains a
   downstream provenance task and does not block the chosen reference. Also document whether the
   public label export should include posterior SD alongside the current mean
   score; posterior uncertainty is available in the saved M2 artifact, so it is
   not missing source data.

9. **Explicit run recipe and acceptance criteria.** Assemble the commands,
   working-directory layout, model choices, and fresh output locations, avoiding
   accidental checkpoint reuse. Keep exact comparisons for deterministic stages;
   define numerical and statistical checks for a fresh M2 fit, including posterior
   summaries, convergence, and ranking stability. This is routine implementation
   work, not missing scientific evidence. Path configuration changes are now
   authorized. The migrated scripts now expose explicit paths and model selection;
   see the validation record for the executed stages.

## Proposed first migration boundary

Start from the eight core candidates and the NSD correction dependency, preserving
the current research workspace. Keep optional diagnostics separate and leave the
unreported benchmark experiments there. Recover the publication inputs and run
settings before treating any cleaned command as a reproduction of a paper result.

The shared interface to Routes A and B should explicitly identify the image,
target, score, and any available uncertainty, together with the label-generation
configuration. The behavioral cell-held-out CV used for model selection is
different from the image-held-out prediction splits used by the routes.
