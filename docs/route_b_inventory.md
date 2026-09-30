# Route B source and reproduction inventory

Inventory date: 2026-09-10. This records recovered source and artifacts, not a
completed reproduction or an endorsement of the historical validation protocol.

Port update, 2026-09-12: the original implementation is now organized in `route_b/`.
The [reproduction guide](route_b_reproduction.md) describes equivalence checks and
the separate fresh-training experiment. Follow-up comparison found that all five
baseline fold states and category means exactly match the named auxiliary-zero
checkpoint, which supplies additional head-training metadata. The original
findings below remain the audit record.

## Provenance and scope

- Source repository: `/home/riccardo/projects/preds`, branch `review`.
- Source commit: `a0d2758613a9dba585b890f4d1615cfcfe4e0b12`.
- Paper: `AIPA_IJCV_26.pdf`, §5.7, Figure 7, Table 7, Figures 8 and 9;
  Route B also contributes predictions to §5.8.
- Machine-readable evidence: [route_b_audit.json](route_b_audit.json), including
  source-file hashes, array shapes, reconstructed split counts, checkpoint
  comparisons, and saved metrics.

The source README describes an older single-split result and is not the
authority for the final experiment. The five-fold checkpoint and saved analysis
metrics match the manuscript's final baseline. Figure 5 is the stimulus
embedding predictor and belongs to Route A.

The audit read source, stored index arrays, feature headers, reports, and
checkpoint tensors on CPU. It reconstructed splits using the recovered split
functions and compared all 44 encoder tensors against each of the five saved
folds. It did not retrain or run full prediction. Audit package versions are
recorded in the JSON; they are not evidence of the original training environment.

A follow-up source and paper check confirmed that the manuscript explicitly
requires outer-training-only PCA, scaling, and target means (printed page 20),
with encoder training nested inside each outer training fold. The normalization
behavior below is also present in source commit `ee84620` from 2026-02-28.
`outputs/logs/04_train_dnn.log` explicitly records exclusion of 999 complexity
validation IDs. `outputs/logs/06_generate_results.log` records five-fold pooled
evaluation, 18,736 observations, and Pearson 0.6125. These logs corroborate the
recovered flow but do not provide a complete, immutable link between every
artifact, its exact training inputs, and a source revision.

## Data lineage

`preds/full_ranking_M2_n_studentT.csv` is byte-identical to the publication
repository's `complexity/reference/nsd_m2_ranking.csv`. SHA256:

```text
2101a2143b00da45625afeaeda08880c70761f825efb0a7c17769624c0a0e6fb
```

The file contains 12,447 records. `load_complexity_records(train_only=True)`
keeps 12,108 records with a `train-` image name, covering 9,990 physical NSD
images, 11,304 distinct image-target pairs, and 16 targets. The task column
supplies the category; the image filename supplies the physical NSD ID.
Duplicates and record order affect training and must not be changed casually.

The fMRI/image root is
`/data/COMMON/brain_activation/algonauts_2023_challenge_data/`. Each subject's
`training_split/` contains `training_images/`,
`training_fmri/lh_training_fmri.npy`, and
`training_fmri/rh_training_fmri.npy`. Subject ROI masks define the stream tiers.
The `train-` index in the image filename, minus one, locates its fMRI row.
A ranking record contributes one sample for every subject with a matching row.

The five validation folds consequently pool 18,736 subject-record observations,
not 18,736 distinct images. Subject counts are 2,422, 2,486, 2,279, 2,156,
2,470, 2,266, 2,480, and 2,177 for subjects 1 through 8.

## Stages and original implementation

Paths in this section are relative to `preds/`.

| Stage | Entry point | Supporting implementation and behavior |
| --- | --- | --- |
| Visual features | `scripts/02_extract_features.py` | `src/features/extract_dino.py`, `extract_clip.py`; DINOv2 image layers, CLIP image vectors, and target text vectors. |
| fMRI preprocessing | `scripts/03_prepare_roi_pca.py` | `src/data/nsd_utils.py`, `roi_utils.py`; stream ROIs, per-subject PCA, and stored 90/10 row indices. |
| Visual pretraining | `scripts/04_train_dnn.py` | `src/training/train_feature_decoder.py`, `src/data/datasets.py`, `src/models/fmri_encoder.py`. |
| Complexity head | `scripts/05_train_complexity_model.py` | `src/training/train_complexity.py`, `src/models/complexity_head.py`, dataset and split helpers. |
| Baseline evaluation | `scripts/06_generate_results.py` | `src/analysis/evaluate.py`; pooled held-out predictions, full/residual metrics, intervals, and figures. |
| Figure 9 report | `scripts/10_generate_routeb_report.py` | Same encoder/head training with visual/reconstruction weights; saved sweep checkpoints and metrics. |
| Ground-truth variance | `scripts/12_compute_complexity_variance.py` | Shared scientific analysis of image-target labels. |
| Predicted variance | `scripts/13_compute_predicted_complexity_variance.py` | Fold inference, subject averaging, and variance decomposition. |

### Visual features and fMRI preparation

DINOv2 uses `dinov2_vitl14`, with 24 raw intermediate CLS vectors of dimension
1,024. The source manually runs the blocks, without applying a final layer norm
to each intermediate vector. It resizes to 256, center-crops to 224, and uses
ImageNet normalization. Layers 0:8, 8:16, and 16:24 are averaged into three
supervision tiers. This backbone differs from ScanDiff's image encoder.

CLIP uses OpenCLIP `ViT-L-14`, pretrained `laion2b_s32b_b82k`, with normalized
768-dimensional image vectors. For each target, the text prompts are:

```text
a photo of a {}.
a photograph of a {}.
the target object is a {}.
a {} in a scene.
a photo containing a {}.
```

Each prompt vector is normalized; their average is normalized again.
Image decode failures in the original extractors become gray images. Existing
feature files are reused based on their presence, without validating settings
or source-image hashes. These behaviors need explicit handling in the port.

Stream labels are 1 for early, 2/3/4 for middle, and 5/6/7 for late. Each
subject/tier PCA fits 2,048 components using a random 90% row split with seed 42.
The three transformed tiers concatenate to 6,144 dimensions. The saved PCA
models and split arrays are part of the reproduction inputs.

### Encoder pretraining and complexity fitting

Subject-specific adapters map 6,144 inputs to 2,048 dimensions. A shared MLP
produces the brain representation. Projection heads predict the three DINO
tiers and CLIP image embedding. Each visual loss combines cosine distance,
MSE, and symmetric InfoNCE with weight 0.5 and temperature 0.07.

The original encoder-training data excludes a fixed 999-image holdout in
addition to applying the per-subject PCA training indices. Training rows
determine its fMRI normalization. The validation dataset constructs its
normalization separately, without applying that same image exclusion.

Phase 3 uses `StratifiedGroupKFold`, five folds, shuffle enabled, seed 42,
target stratification, and physical NSD image grouping. Category means are
computed from training ranking records before expansion to available subjects.
The target is the score minus its category mean. A subject/category-balanced
replacement sampler feeds training batches.

The head combines a learned 32-dimensional target vector with CLIP text
conditioning. FiLM modulates the projected brain representation. The head
outputs a residual mean and log variance, with log variance clamped to [-7, 2]
for the Gaussian negative log likelihood.

All encoder tensors in all five saved fold checkpoints exactly equal the
pretrained encoder tensors. The encoder was therefore unchanged in these
artifacts. Source defaults are 100 encoder-pretraining epochs and 60 head
epochs, AdamW with learning rate 0.0003, weight decay 0.01, and patience 15.
These are recovered defaults, not verified original run arguments: neither
baseline checkpoint stores a complete run configuration. The pretrained best
epoch is 75; the saved best complexity fold is fold 3.

### Evaluation and related experiments

Baseline evaluation concatenates each fold's held-out predictions. The code
calls this an ensemble, but does not average five models per observation.
It adds the saved category mean to the residual prediction and reports
`sigma = exp(0.5 * log_variance)`. It also computes 1,000-bootstrap confidence
intervals and 5,000-permutation tests.

Figure 9 varies the encoder visual-loss weight over 0, 0.25, 0.5, 0.75, and 1;
the fMRI reconstruction weight is its complement. Each point then uses the
frozen-encoder complexity flow. The four additional sweep runs disable the
auxiliary visual loss during head training. The visual-only endpoint reuses
the original baseline checkpoint.

| Visual weight | Full-score Pearson | Residual Pearson |
| --- | --- | --- |
| 0 | 0.4994 | 0.0458 |
| 0.25 | 0.5939 | 0.3607 |
| 0.5 | 0.6106 | 0.3953 |
| 0.75 | 0.6086 | 0.3917 |
| 1 | 0.6125 | 0.3995 |

For predicted variance, script 13 keeps one prediction per physical
image-target pair and subject, using that image's fold model, then averages
subjects. Repeated label records are handled differently here than in head
training. The earlier complexity validation recovered the reported 1,173-image
prediction analysis, with within-image variance 0.007898, or 30.82% of total
variance. The label analysis uses 1,205 images and gives 47.16%. See the
[complexity validation record](complexity_validation.md).

## Identified artifacts

These paths are relative to `preds/` and remain external to the clean repository.

| Artifact | Role |
| --- | --- |
| `outputs/features/dino/{subject}_dino_layers.npy` | Subject-ordered arrays of shape `(N, 24, 1024)`. |
| `outputs/features/clip_image/{subject}_clip_img.npy` | Subject-ordered arrays of shape `(N, 768)`. |
| `outputs/features/clip_text/clip_text_embeddings.npz` | Target text vectors. |
| `outputs/features/pca/{subject}_pca_fmri.npy` | Subject-ordered arrays of shape `(N, 6144)`. |
| `outputs/features/pca/{subject}_pca_models.pkl` | Fitted PCA transforms. |
| `outputs/features/pca/{subject}_train_idx.npy`, `{subject}_val_idx.npy` | Stored subject row partitions. |
| `outputs/checkpoints/feature_decoder/dnn_multisubj_best.pt` | Baseline pretrained encoder and feature heads. |
| `outputs/checkpoints/complexity/multisubj_best.pt` | Five fold model states, metrics, and one category-mean dictionary. |
| `outputs/analysis_metrics.json`, `outputs/results_summary.txt` | Table 7 and per-subject Figure 8 results. |
| `outputs/routeb_phase3_sweep_metrics.json` | Figure 9 objective sweep results. |
| `outputs/figures/routeb_summary/phase3_spectrum_full_vs_residual.png` | Saved sweep figure. |

The stored baseline reports Pearson 0.61245578, Spearman 0.56750648,
MAE 0.17522320, R² 0.37293759, residual Pearson 0.39945856,
residual Spearman 0.37925900, and mean sigma 0.20171666. These round to Table 7.
The saved per-subject results match Figure 8.

## Protocol discrepancies requiring an explicit decision

These findings concern Route B. They are separate from the previously deferred
COCO scanpath-label correction.

1. **Pretraining and PCA are not fitted separately for each complexity fold.**
   The five complexity partitions themselves have zero physical-image overlap
   between training and validation. However, they reuse one pretrained encoder
   and the same subject PCA transforms. The recovered code excludes a fixed
   999-image set during encoder training, rather than each fold's held-out set.
   Reconstructing membership from the source and saved row indices gives the
   following counts of held-out physical images present in fitting data for at
   least one subject:

   | Fold | Held-out images | Encoder fitting overlap | PCA fitting overlap |
   | --- | --- | --- | --- |
   | 1 | 1,996 | 1,606 | 1,789 |
   | 2 | 2,001 | 1,616 | 1,798 |
   | 3 | 1,996 | 1,617 | 1,787 |
   | 4 | 1,998 | 1,626 | 1,806 |
   | 5 | 1,999 | 1,645 | 1,821 |

   These counts describe the recovered code and stored indices. The baseline
   checkpoint does not itself store the historical pretraining image IDs.
   Encoder tensor equality verifies that the same encoder is reused, but does
   not independently prove every original fitting input.
2. **Phase 3 normalization uses all subject rows.** `ComplexityDataset`
   computes each subject's mean and standard deviation from the full PCA array,
   not that complexity fold's training rows. This differs from the paper's
   train-only preprocessing description and from Phase 2 normalization.
3. **Evaluation reuses one fold's category means.** Training computes a
   dictionary per fold, but the baseline checkpoint retains only the best
   fold's dictionary, from fold 3. Evaluation uses it to construct residual
   labels and restore full scores in all five folds. Other folds' training
   means differ by up to 0.03088. Saving and applying means per fold would
   change the reported predictions and residual metrics.
4. **Original run metadata is incomplete.** Baseline checkpoints omit full
   run arguments, fold-specific preprocessing and means, split identifiers,
   and environment information. Existing source defaults and later experiment
   reports cannot establish every original baseline setting.

A faithful replay of the existing artifacts and an experiment implementing the
paper's stated fold-specific protocol are different validation targets. Porting
must keep this distinction explicit. Correcting the issues above would require
new predictions and, for fold-specific PCA/pretraining, new training runs.

## Port boundaries and next steps

The first implementation pass should preserve the source-to-file mapping and
checkpoint compatibility, while making paths configurable. A proposed layout is:

```text
route_b/
├── README.md
├── data/           # NSD record loading, subject alignment, ROI/PCA preparation
├── features/       # DINOv2 and CLIP extraction
├── models/         # fMRI encoder and target-conditioned head
├── training/       # Encoder pretraining and complexity fitting
├── evaluation/     # Prediction, metrics, and paper reports
└── experiments/    # Baseline and Figure 9 configurations
```

`src/config.py` currently creates directories when imported and embeds local
paths. Replace this setup deliberately during porting. Do not copy the large
arrays or checkpoints into Git. The first execution check can compare inference
from saved checkpoints before attempting any fresh training. Prediction can
avoid loading image embeddings: script 13 already demonstrates this separation,
whereas the general dataset class loads them even during evaluation.

The following files are not additional mandatory stages of the core flow:

- `scripts/01_build_nsd_coco_mapping.py` prepares metadata mappings which the
  current core training/evaluation code does not consume.
- `scripts/07_plot_architecture.py`, `08_data_audit.py`, and
  `09_summarize_ablation_checkpoints.py` support presentation and inspection.
- `scripts/11_generate_dino_clip_ablation_report.py` is a separate feature-loss
  follow-up; it is not the visual/reconstruction sweep used in Figure 9.
- Scripts 12 and 13 mix Route B prediction with cross-route analysis. Keep
  model-specific prediction in `route_b/` and the shared decomposition in
  `analysis/`.

Before a full training reproduction, recover the available baseline launch
records, pin a dedicated environment, and decide how the historical protocol
and any corrected protocol will be represented. The running complexity jobs
use their existing source snapshot and are independent of this inventory.
