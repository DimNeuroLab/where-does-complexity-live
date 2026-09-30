# Known scientific flaws and unresolved evidence

The reproduction target is the existing `AIPA_IJCV_26.pdf` manuscript. This document distinguishes confirmed
historical behavior from uncertain provenance. Historical reproduction and later methodological corrections
are separate experiments. Reproducing a computation does not establish its statistical validity.

## Complexity

### C-1. Response-time table provenance differs from the captions

**Status and evidence:** Strongly supported mismatch, with incomplete original execution evidence.
Tables 1 and A1 describe human response times. Recovered summaries match 112 of 117 printed cells,
including all 27 LOO cells. Historical commands, a 2,779-image fingerprint and targeted M1/M2 fits instead
support unfiltered ScanDiff predictions: 31,010 trials, including failures and unknown success flags.
Actual human inputs have 24,880 valid RT trials and 2,241 filename identities. The exact deleted input,
full original posteriors and complete execution manifest were not recovered. The remaining seven models
and original five-fold CV have not been independently reproduced on the candidate input. Five printed
cells also differ from the recovered summaries.

**Consequences:** Comparing an actual human run with those tables changes the experiment. The discrepancy
is not explained by a small numerical tolerance. The two candidate fits support provenance, not complete
historical replay. **Historical behavior:** retain both explicitly named input flows and the original model
definitions; do not label synthetic input as human. **Deferred remedy:** recover the remaining evidence or
revise the scientific account and tables together. See [RT models](complexity/models/fit_response_time.py)
and the [component workflow](complexity/README.md). The external September 14 RT provenance audit preserves
the recovered tables, source identities, commands and numerical comparisons.

### C-2. Original joint CV pairs parameters from different posterior draws

**Status and evidence:** Confirmed original scoring error. The original scorer concatenates predicted
means in blocks containing every posterior draw, but expands shape and scale parameters with `repeat`.
Matching those blocks requires `tile`. The current [RT CV code](complexity/models/fit_response_time.py)
already uses the corrected expansion.

**Consequences:** Joint predictive ELPD and RMSE can change. Fitted posteriors and ranking exports are
unaffected. The full historical effect is unknown without the original fold posteriors. **Historical
behavior:** an explicit historical reproduction path must retain the original pairing; the corrected
scorer belongs to a separate path. That separation still requires implementation. **Deferred remedy:**
use aligned pairing for corrected experiments and report new scores.

The [runner's joint LOO preparation](complexity/models/runner.py) is a separate issue: it restores an
original marginalization step omitted during porting. Keeping that repair recovers original behavior.
Its saved-posterior rescoring does not validate the original joint CV pairing.

### C-3. Selection, image identity, units and RMSE estimands differ

**Status and evidence:** Confirmed historical choices, with manuscript ambiguities. Human trials are
selected by image eligibility without requiring every retained trial to be successful; generated count
trials use a success filter. Count models discard zero saccades. The converter reconstructs per-subject
trial order after selection, so changed success flags can change the trial covariate for otherwise
unchanged scanpaths. Image effects use filename identity. COCO target category is not part of that model
key, and ranking export maps each filename to its last encountered task. RT remains in milliseconds, so
the shifted model's prior scale of 0.3 has that unit. M1/M2 CV uses mean log RT, whereas M3/M4 uses the log
of predictive mean RT.

**Consequences:** Filtering, regrouping targets, converting units or standardizing point predictions changes
the experiment. Their combined numerical effect is not established. **Historical behavior:** preserve
these choices. **Deferred remedy:** evaluate corrected selection, image-target identities and explicit
units and estimands separately. See the [converter](complexity/preprocessing/convert_scanpaths.py),
[image eligibility filter](complexity/preprocessing/select_coco_trials.py),
[RT models](complexity/models/fit_response_time.py), [count models](complexity/models/fit_movement_count.py)
and [ranking export](complexity/exports/export_rankings.py).

### C-4. Figure references and baseline inputs are inconsistent

**Status and evidence:** Confirmed numerical reference mismatch. The historical mean-N comparison uses
successful generated counts and gives Spearman 0.701049 against human M2 RT. Figure 4a's correlation
0.711098 and hardest-20 overlap 0.35 agree with a human count reference. Using the stated RT reference
gives 0.655243 and 0.20.

**Consequences:** Changing the reference ranking materially changes the comparison. **Historical behavior:**
keep the original input rankings explicit. **Deferred remedy:** align the caption, reference identity and
interpretation. See [mean count](complexity/models/mean_count_baseline.py) and
[ranking comparisons](complexity/evaluation/compare_ranking_pair.py).

### C-5. Completion and table agreement do not establish convergence

**Status and evidence:** Confirmed diagnostic failures in historical and corrected candidates. The later
26-model comparison completed every job, but only four models passed its combined sampling checks.
Its nine-model unfiltered sensitivity has two passes. Alternative likelihoods and joint models exhibit
poor ESS, mixing or divergences in several populations. Opposed chain modes are documented for M2J.
Posterior sampling acceptance and PSIS-LOO reliability are separate checks.

**Consequences:** A finished job or rounded table match cannot establish a valid model comparison.
**Historical behavior:** retain original outputs with their diagnostic status; do not replace them with
newer corrected outputs merely because those exist. **Deferred remedy:** investigate identifiability,
sampling and scoring in a separate corrected study. See the [model runner](complexity/models/runner.py)
and [corrected comparison](complexity/evaluation/run_corrected_comparison.py).

### C-6. Fresh NSD scanpaths do not preserve label membership

**Status and evidence:** Confirmed reproduction limitation. Saved scanpaths reproduce the fixed 12,447-row
ranking exactly. Fresh generation has 14 added and 13 missing labels, despite common-label Spearman
0.999399 and RMSE 0.010182. Small coordinate and success changes cross inclusion boundaries; one qualifying
trial can determine whether a label exists. The exact historical environment or arithmetic cause is unknown.

**Consequences:** Strong aggregate agreement does not establish identical supervision for either route.
**Historical behavior:** use verified historical labels for A/B reproduction and retain the fresh identity
failure. **Deferred remedy:** recover the original environment or explicitly define a new dataset version.
See [NSD correction](complexity/preprocessing/correct_nsd_targets.py) and
[ranking verification](complexity/evaluation/verify_ranking.py).

## Route B

### B-1. Preprocessing and pretraining are not nested inside the head folds

**Status and evidence:** Confirmed recovered behavior, inconsistent with the manuscript's description of
preprocessing fitted only on outer training data. Subject PCA uses stored 90% row partitions, and one
pretrained encoder is reused across all five head folds. Encoder training excludes a fixed 999-image
holdout, rather than each outer validation fold. Reconstructed held-out overlap ranges from 1,606 to 1,645
images for encoder fitting and 1,787 to 1,821 for PCA fitting. Checkpoints prove encoder weight reuse but
do not independently identify every historical fitting image.

Phase 2 constructs validation normalization separately from training normalization. Phase 3 computes
subject means and scales from all PCA rows. **Consequences:** These metrics do not establish performance
under the fully nested protocol described in the paper. **Historical behavior:** preserve the original
PCA, pretraining and normalization populations. **Deferred remedy:** evaluate fold-specific preprocessing
and pretraining separately. See [datasets](route_b/data/datasets.py), [ROI/PCA](route_b/data/roi.py) and
[encoder training](route_b/training/encoder.py).

### B-2. Frozen parameters retain dropout; evaluation reuses one fold's means

**Status and evidence:** Confirmed historical behavior. Calling `model.train()` leaves encoder dropout
active while its parameter gradients are disabled. All 44 saved encoder tensors equal the pretrained
encoder in all five folds. Training computes fold-specific category means, but the baseline checkpoint
retains the best fold's means, from fold 3, and applies them to all evaluation folds. Other means differ
by up to 0.03088.

**Consequences:** Forcing encoder evaluation mode during training or using means from each fold changes
the experiment. The isolated effect of each change is not established. **Historical behavior:** retain
dropout and saved-mean semantics. **Deferred remedy:** store explicit fold-specific state and statistics
in corrected experiments. See [head training](route_b/training/complexity.py) and
[evaluation](route_b/evaluation/metrics.py).

### B-3. Historical metadata and feature arithmetic have limits

**Status and evidence:** Original encoder launch metadata is incomplete; some settings come from source
defaults. Extraction is sensitive to batch shape: DINO at the original batch size of 64 matches exactly;
CLIP at 128 matches within 1e-6, while a four-image batch differs more. DINO raw block CLS vectors, source
revision and CLIP prompts and weights are part of the experiment.

**Consequences:** Generic backbone or batch substitutions can change supervision despite matching feature
dimensions. **Historical behavior:** preserve known models, transforms, batches and settings. **Deferred
remedy:** capture complete provenance for future runs without inventing missing historical metadata.
See [DINO extraction](route_b/features/dino.py), [CLIP extraction](route_b/features/clip.py) and
[workflow](route_b/README.md).

## Engineered Route A

### A1-1. The port and recovered notebook describe different recipes

**Status and evidence:** Confirmed consequential differences. The COCO notebook drops only bowl, retaining
2,134 rows and 17 categories; PR #1 also drops 110 knife rows. The notebook combines DINO-small PCA-64,
26 detector features and 17 task features. PR #1 adds CLIP PCA-64 and changes squared error, 4,000 trees
and depth 5 to absolute error, 300 trees and depth 6. Row-based CV and preprocessing become image-grouped
folds with preprocessing fitted within each fold.

**Consequences:** Predictions and metrics can change materially; numerical equivalence is not established.
**Historical behavior:** actual original recipes must remain distinct from generic prediction functionality.
**Deferred remedy:** restore experiment-specific recipes from evidence before claiming fidelity. Sources:
[original notebook](https://github.com/DimNeuroLab/aipa/blob/970c95f180ac1b2befabd656f2437c3416d3b5db/image_to_complexity_coco.ipynb),
[PR #1 training](https://github.com/DimNeuroLab/where-does-complexity-live/blob/94cf8e2e26ac07bee34d80c36dcf2327247df66d/route_a/engineered/training.py)
and [datasets](https://github.com/DimNeuroLab/where-does-complexity-live/blob/94cf8e2e26ac07bee34d80c36dcf2327247df66d/route_a/engineered/datasets.py).

### A1-2. Final engineered and fMRI ablation provenance is missing

**Status and evidence:** Unresolved. Tables 4, 5 and B1 report 12,476 samples, OpenCLIP and residual fMRI
PLS16. The recovered NSD notebook reports 9,137 stimulus rows and 2,190 fMRI rows on six subjects and lacks
the final PLS16 recipe. Its feature CSVs and the final table-producing source were not found in the
examined branches and workspaces. The user does not know their location. PR #1 does not implement the
full feature grid or fMRI augmentation.

**Consequences:** These paper results cannot presently be claimed reproduced. **Historical behavior:**
retain known original experiments with their actual populations; do not invent a matching experiment.
**Deferred remedy:** recover source, features, labels and folds, or explicitly retain the unresolved result.
See the [recovered NSD notebook](https://github.com/DimNeuroLab/aipa/blob/970c95f180ac1b2befabd656f2437c3416d3b5db/image_to_complexity_ablation_nsd.ipynb)
and [PR #1 scope](https://github.com/DimNeuroLab/where-does-complexity-live/blob/94cf8e2e26ac07bee34d80c36dcf2327247df66d/route_a/engineered/README.md).

### A1-3. Inventory labels are not interchangeable with ranking records

**Status and evidence:** Confirmed population and formatting distinctions; final paper-input provenance
remains unknown. The common ranking has 12,447 rows. Training selection gives 12,108 records and 9,990
physical images. Subject expansion gives 18,736 observations, or 12,476 unique subject-image-target keys.
The inventory generator collapses repeated physical pairs using the last record and six-decimal rounding.
The clean saved inventory matches that rule exactly. The embedding branch's other file, `full_ranking.csv`,
has the same keys but 12,447 of its 12,476 labeled scores are malformed text.

**Consequences:** The difference of 29 does not mean 29 additional images without fMRI. Replacing the full
ranking with the inventory changes observation weights and labels. The malformed copy cannot supply numeric
labels. **Historical behavior:** preserve each experiment's joins and repeated records, including conflicting
scores. **Deferred remedy:** identify the final engineered label artifact; do not silently repair malformed
strings. See the [inventory generator](https://github.com/Crystal-Spider/preds/blob/a0d2758613a9dba585b890f4d1615cfcfe4e0b12/scripts/08_data_audit.py)
and [PR #1 loader](https://github.com/DimNeuroLab/where-does-complexity-live/blob/94cf8e2e26ac07bee34d80c36dcf2327247df66d/route_a/engineered/datasets.py).

## Embedding Route A

### A2-1. Row folds share images and use global category means

**Status and evidence:** Confirmed inherited behavior. The dataset retains all 12,108 training records,
despite a docstring describing unique pairs. Shuffled row folds put 476, 480, 454, 487 and 482 physical
images on both sides of training and validation. Category means are computed from all records before folding.

**Consequences:** This is not the common image-grouped outer CV described in the cross-route manuscript
comparison. **Historical behavior:** retain original row folds, record order and global means.
**Deferred remedy:** evaluate grouped folds and training-only means in a new study. See
[PR #2 model](https://github.com/DimNeuroLab/where-does-complexity-live/blob/174c84cce03686adc665cab4890a7342d0cbf1e8/route_a/embedding/model.py)
and [training](https://github.com/DimNeuroLab/where-does-complexity-live/blob/174c84cce03686adc665cab4890a7342d0cbf1e8/route_a/embedding/training.py).

### A2-2. Saved state and best-epoch training metrics differ

**Status and evidence:** Confirmed inherited behavior. Early stopping tracks the best loss and correlation
but never restores the corresponding weights. A checkpoint therefore contains final-epoch weights while
its metadata describes the best epoch. Saved pooled evaluation for Table 6 is a separate statistic from
the training summary of best fold correlations.

**Consequences:** A best-weight correction changes the historical state. Its numerical effect is unknown
without original epoch states. **Historical behavior:** preserve final-epoch checkpoint semantics and
explicit metric meanings. **Deferred remedy:** store and restore best weights in a corrected training path.
See [original training](https://github.com/DimNeuroLab/aipa/blob/e29731ee7a2dd8972ffa61b7e0e32a1d0c2cc5a7/scripts/9_train_route_a_ablation_emb_kfold.py)
and [PR #2 training](https://github.com/DimNeuroLab/where-does-complexity-live/blob/174c84cce03686adc665cab4890a7342d0cbf1e8/route_a/embedding/training.py).

### A2-3. Interval labels and incomplete evaluation

**Status and evidence:** Confirmed inherited issues. Global evaluation calls a Pearson bootstrap routine
for outputs labeled as Spearman confidence intervals. Evaluation can skip folds without `val_idx` and
aggregate the remaining predictions.

**Consequences:** Those intervals are not established Spearman intervals. A completed command need not
cover the entire evaluation population. **Historical behavior:** retain historical numeric outputs with
these qualifications; reproduction validation must still require complete fold and sample coverage.
**Deferred remedy:** use a rank-based bootstrap and explicit fold validation in a separate correction.
See [evaluation](https://github.com/DimNeuroLab/where-does-complexity-live/blob/174c84cce03686adc665cab4890a7342d0cbf1e8/route_a/embedding/evaluation.py).

### A2-4. Feature-only ablations lack a complete source-to-result chain

**Status and evidence:** Unresolved. PR #2 cites earlier per-ablation scripts, but the frozen branch contains
combined-model source and metrics, not the original DINO-only and CLIP-only scripts, checkpoints and results.
The new flags alone do not establish reproduction of Figure B1. Table B1 belongs to engineered and fMRI
ablations, not this model.

**Consequences:** Main-model definition equivalence cannot validate feature-family results or recover trained
weights. **Historical behavior:** retain the combined small/noise-zero recipe with saved Table 6 and Figure 6
evidence. **Deferred remedy:** recover feature-specific artifacts and correct paper references. See the
[PR #2 README](https://github.com/DimNeuroLab/where-does-complexity-live/blob/174c84cce03686adc665cab4890a7342d0cbf1e8/route_a/embedding/README.md).

## Cross-route analyses

### X-1. Higher-threshold variance populations are unresolved

**Status and evidence:** The historical K>=2 decomposition is supported; higher-threshold ground-truth
provenance is unresolved. The full ranking gives 1,205 images at K>=2, total variance 0.07223498, within
variance 0.03406886 and within share 47.16394%. At K>=3 it gives 132 images and 61.74275%; at K>=4,
13 images and 88.36508%. Training-only labels give 129/12 images and 61.01846%/86.91923%. The paper states
117/12 images and 60.0%/80.5%. Exactly three training targets explains 117 images but not the variance
values. Last-record aggregation does not jointly recover the table either. B predictions use 1,173 images
at K>=2 and 129/12 at higher thresholds, with shares 30.8198%, 42.3319% and 78.49999%. A's original variance
source and predictions were not recovered.

**Consequences:** Population and duplicate aggregation affect variance and recovery ratios. A single common
population behind every Table 8 row is not established. **Historical behavior:** retain the original
average-of-repeats decomposition with explicit population and threshold. Do not change joins to force
agreement. **Deferred remedy:** recover the exact higher-threshold ground truth and A prediction artifacts,
then reconcile all rows. See [label variance](complexity/evaluation/compute_variance.py) and
[B prediction variance](route_b/evaluation/variance.py).

### X-2. Similar modules encode different experiments

**Status and evidence:** Confirmed source differences. Engineered A uses DINO-small; embedding A and B use
DINO-large; ScanDiff uses DINO-base with registers. A's small model uses a 16-dimensional category embedding
and dropout 0.5; B uses 32 dimensions and 0.4. PR #2's shared head default of 0.2 and unpinned feature
loading differ from the validated B implementation.

**Consequences:** Standardizing imports or defaults can silently change the backbone, model state or
arithmetic. **Historical behavior:** retain experiment-specific parameters and feature protocols.
**Deferred remedy:** share implementations only after explicit equivalence checks, preserving intentional
differences. See [B models](route_b/models/complexity.py), [B features](route_b/features/dino.py) and
[PR #2 shared head](https://github.com/DimNeuroLab/where-does-complexity-live/blob/174c84cce03686adc665cab4890a7342d0cbf1e8/shared/complexity_head.py).
