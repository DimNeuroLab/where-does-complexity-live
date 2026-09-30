# Route B: neural readout

## Purpose and paper mapping

Route B predicts target-conditioned complexity from NSD fMRI, subject identity and a search target.
An encoder learns visual representations from brain responses; a conditional head learns the
[complexity labels](../complexity/README.md). This corresponds to Section 5.7, Figure 7, Table 7,
per-subject Figure 8, the objective sweep in Figure 9 and prediction-side variance analysis in Section 5.8.
NSD responses were recorded during image viewing; the search target is an additional model input.

## Workflow

1. Extract DINOv2 and CLIP image features and CLIP target text vectors.
2. Concatenate left/right fMRI hemispheres and group stream ROIs into early `[1]`, middle `[2,3,4]` and
   late `[5,6,7]` tiers. Fit up to 2,048 PCA components per tier and subject, then transform all rows.
3. Pretrain subject-specific adapters and a shared 2,048-dimensional encoder using visual targets and
   optional fMRI reconstruction. Reuse this encoder across the complexity folds.
4. Fit five target-conditioned heads with physical-image-grouped, category-stratified folds.
   Encoder parameters are frozen while encoder dropout remains active during head training.
5. Restore the saved category means, export held-out predictions and compute pooled, subject and category metrics.
6. Repeat the four additional objective weights and decompose variance after averaging repeated subject/pair predictions.

The fixed training ranking contains 12,108 records across 9,990 physical images and 11,304 physical
image-target pairs. Joining each record to every subject who viewed its NSD image gives 18,736 observations.
Repeated records are retained. These yield 12,476 distinct subject-image-target keys; this is a separate
counting unit from the 12,447-row full ranking.

## Inputs, identities and sources

Use the [Algonauts 2023 challenge package](https://algonautsproject.com/2023/challenge.html#challenge-data),
with its [official tutorial](https://github.com/gifale95/algonauts_2023). It packages the
[Natural Scenes Dataset](https://www.naturalscenesdataset.org/) images, responses and ROI masks consumed here.
The images originate in [COCO](https://cocodataset.org/#download).

```text
NSD_ROOT/
  subj01/                              # Repeat through subj08
    training_split/
      training_images/train-0001_nsd-00013.png
      training_fmri/lh_training_fmri.npy
      training_fmri/rh_training_fmri.npy
    roi_masks/lh.streams_challenge_space.npy
    roi_masks/rh.streams_challenge_space.npy
```

The first filename index is one-based and identifies the subject's fMRI row. The NSD ID identifies the
physical image across subjects. Keep filenames and row order. Subject image counts are 9,841, 9,841,
9,082, 8,779, 9,841, 9,082, 9,841 and 8,779. Hemisphere columns and ROI masks use challenge vertex order.
Use the challenge's session-standardized responses averaged across repeated presentations.

The default labels are the installed [complexity/nsd_m2_ranking.csv](../complexity/nsd_m2_ranking.csv),
with columns `image,score,task`. Labels, repeated records and score precision remain unchanged.
The 16-category order is in [constants.py](constants.py).

### Backbone identities

| Backbone | Source and representation |
| --- | --- |
| DINOv2 | `dinov2_vitl14`, ViT-L/14 without registers; [source revision](https://github.com/facebookresearch/dinov2/tree/7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c). |
| CLIP | OpenCLIP `ViT-L-14`, `laion2b_s32b_b82k`; [weight revision](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/tree/1627032197142fbe2a7cfec626f4ced3ae60d07a). |

DINO uses RGB, bicubic shorter-edge resize to 256, center crop 224 and ImageNet normalization.
Retain raw CLS tokens after all 24 blocks, before final backbone normalization; three groups of eight
blocks produce 1,024-dimensional tier averages. Its extraction batch size is 64.
CLIP uses its pretrained transform, L2-normalized 768-dimensional image vectors and batch size 128.
Text extraction normalizes each of five prompt vectors, averages them, and normalizes again.
[features/clip.py](features/clip.py) contains the exact templates and pinned loading procedure.

| Weight file | SHA-256 |
| --- | --- |
| [DINOv2 ViT-L/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) | `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428` |
| CLIP `open_clip_pytorch_model.bin` at the revision above | `5ddb47339f44e4fd9cace3d3960d38af1b51a25857440cfae90afc44706d7e2b` |

The extractors substitute gray images for unreadable files. Keep the input image tree complete and retain
these transforms and batch sizes when generating scientific features.

## Installation

From the repository root, use a separate Python 3.12 environment:

```bash
python3.12 -m venv .venv-routeb
source .venv-routeb/bin/activate
python -m pip install -c route_b/requirements.lock.txt '.[route-b]'
```

[requirements.lock.txt](requirements.lock.txt) records PyTorch 2.10.0/CUDA 12.8, OpenCLIP 3.2.0,
NumPy 2.4.2 and scikit-learn 1.8.0. Training and full evaluation use CUDA. Standalone row prediction
also supports CPU. The training implementation computes on the first configured device.
Full feature banks and checkpoints require substantial external storage; PCA can use large amounts of host RAM.

## Commands

Set data, output and model-cache locations, then copy [paper.example.json](experiments/paper.example.json):

```bash
export NSD_DATA=/path/to/algonauts_2023_challenge_data
export ROUTE_B_OUTPUT=/path/to/new/route-b-run
export TORCH_HOME=/path/to/model-cache/torch
export HF_HOME=/path/to/model-cache/huggingface
cp route_b/experiments/paper.example.json /path/to/route-b.json
python -m route_b download-models
python -m route_b run --config /path/to/route-b.json
```

`download-models` fetches the pinned weights into library caches and writes their actual hashes alongside
those caches. DINO source loads from the pinned Torch Hub revision; optional `dino_root` selects an explicit
local source tree at that revision. These weight caches are separate from extracted-feature output.

The default run stages are `features prepare pretrain train evaluate sweep variance`. `--weight` selects
`0`, `0.25`, `0.5`, `0.75` or `1` for an individual run; the default is `1`. `sweep` adds the four other weights.
Stages can be selected in separate invocations of the same configuration:

```bash
python -m route_b run --config /path/to/route-b.json --stages features prepare
python -m route_b run --config /path/to/route-b.json --stages pretrain train evaluate --resume
python -m route_b run --config /path/to/route-b.json --stages sweep variance --resume
```

JSON-relative paths resolve beside the configuration file after environment-variable and `~` expansion.

| Configuration field | Use |
| --- | --- |
| `output_dir` | Writable run directory. |
| `inputs.nsd_root` | Challenge input tree. |
| `inputs.ranking_file` | Optional ranking override; defaults to the packaged canonical ranking. |
| `inputs.features_dir` | Optional existing feature tree; otherwise `output_dir/features`. |
| `inputs.pca_dir` | Optional existing PCA observation directory; otherwise `output_dir/pca`. |
| `inputs.pca_models_dir` | Optional saved PCA parameters for `transform`. |
| `inputs.checkpoints_dir` | Optional existing model directory; otherwise `output_dir/checkpoints`. |
| `dino_root` | Optional local DINOv2 source checkout. |
| `head.cv_folds` | Five folds for this workflow. |
| `devices`, `gpu_hours`, `cpu_threads` | Device visibility, persistent GPU-hour allowance and CPU thread limit. |

Generation stages write inside `output_dir`; external feature/PCA/checkpoint inputs are read-only.
For a new evaluation of saved models, point `inputs.pca_models_dir` and `inputs.checkpoints_dir` at their
external directories, keep a fresh output directory, and run `--stages text transform evaluate variance`.
`transform` applies saved PCA parameters without fitting. `text` alone needs no NSD data.

### Prediction from aligned fMRI rows

```bash
python -m route_b predict --checkpoint-file "$ROUTE_B_OUTPUT/checkpoints/complexity/visual_1.pt" \
  --pca-dir "$ROUTE_B_OUTPUT/pca" --nsd-root "$NSD_DATA" --subject subj01 \
  --lh-file /path/to/lh-rows.npy --rh-file /path/to/rh-rows.npy --tasks-file /path/to/targets.json \
  --clip-text-file "$ROUTE_B_OUTPUT/features/clip_text/clip_text_embeddings.npz" \
  --fold 1 --device cpu --output-file /path/to/predictions.csv
```

Hemisphere arrays are two-dimensional and row-aligned. `targets.json` is a JSON list with one canonical
category per row. The PCA directory contains `{subject}_pca_models.pkl` and `{subject}_pca_fmri.npy`;
prediction derives normalization from those stored rows. Image feature arrays are unnecessary at inference.
Use the assigned held-out fold when evaluating paper observations. The model supports its eight trained
subject adapters. `route-b` is the equivalent console command.

For configurable multi-target variance, the utility module accepts `--config`, `--checkpoint-file` and
`--min-pairs`; see `python -m route_b.evaluation.variance --help`.

### Long runs and resume

Use a persistent terminal session or your own job scheduler with an absolute interpreter/configuration path.
Keep run settings fixed. `--resume` checks source, resolved settings, package versions and input fingerprints;
a changed source revision requires a separate run directory. Retain the original source environment to
resume its runs.

Training saves model, optimizer, scheduler, early-stopping and RNG states each epoch. An interrupted partial
epoch repeats from the last saved epoch. Completed stages and PCA subjects are reused after fingerprint
verification. The GPU-hour budget survives restarts; CPU PCA is excluded. Exit code 75 means the budget
has paused execution. Read `status.json`, `budget.json` and `logs/` before resuming.

## Outputs

| Location below `output_dir` | Contents |
| --- | --- |
| `features/dino`, `features/clip_image`, `features/clip_text` | Subject image arrays and the target text NPZ. |
| `pca/` | Subject PCA models, transformed rows and preparation split indices. |
| `checkpoints/feature_decoder/visual_*.pt` | Pretrained encoders. Fractional weights use underscores in checkpoint names. |
| `checkpoints/complexity/visual_*.pt` | Five head states, category means and training metadata. |
| `results/visual_*/` | Fold NPZs, `predictions.csv` and pooled/per-subject/per-category `metrics.json`. |
| `results/variance.json` | Prediction variance at the default two-pair threshold. |
| `summary.json`, `summary.md`, `summary.png`, `summary.pdf` | Summaries and plots of available current-run results. |
| `recovery/`, `inputs/`, `completed/`, `manifest.json`, `logs/` | Epoch recovery, fingerprints, completion records and provenance. |

Prediction exports include record index, NSD ID, fold, subject/category indices, full and residual values,
category mean and log variance. The standalone prediction CSV contains row, subject, target, fold,
complexity and sigma.

## Paper targets and reproducibility settings

Table 7's baseline targets are Pearson 0.6125, Spearman 0.5675, MAE 0.1752, R-squared 0.3729,
residual Pearson 0.3995, residual Spearman 0.3793 and mean sigma 0.2017, on 18,736 observations.

| Retained CSV | Schema and provenance |
| --- | --- |
| [paper_metrics.csv](paper_metrics.csv) | One row per visual weight: full/residual correlations, MAE, R-squared, mean sigma and observation count. Saved paper-checkpoint pooled metrics underlying Table 7/Figure 9. |
| [paper_subject_metrics.csv](paper_subject_metrics.csv) | The same metrics plus `subject`, for each weight/subject; saved paper-checkpoint subject summaries. |
| [paper_variance.csv](paper_variance.csv) | `total,within,between,images,within_percent`; saved baseline prediction decomposition at two or more target pairs per physical image. |

The variance reference has 1,173 images and within-image share 30.8198%; variance values are in squared
complexity-score units. Files retain their original numeric precision.

PCA uses a seeded 90% subject-row partition. Encoder pretraining excludes a fixed 999-image complexity
holdout, and constructs training/validation normalization separately. Head datasets normalize using all
subject PCA rows. Head folds use physical-image grouping and seed 42; training category means are
fold-specific, while pooled checkpoint evaluation uses the single saved category-mean lookup.

The example uses seed 42, batch size 256, AdamW learning rate 0.0003 and weight decay 0.01, patience 15,
maximum 100 encoder epochs and 60 head epochs, encoder freeze duration 100 and zero head auxiliary weight.
Baseline pretraining uses visual weight 1 and reconstruction weight 0; the objective sweep uses the
complementary weights. Loss definitions and zero-weight forward computations remain in their owning
[encoder](training/encoder.py) and [head](training/complexity.py) training modules.
