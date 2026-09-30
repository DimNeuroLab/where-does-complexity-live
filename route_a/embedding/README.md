# Route A: embedding-conditioned stimulus readout

## Purpose and paper mapping

This variant predicts complexity directly from image embeddings and search-target conditioning.
It corresponds to Section 5.6.1, Figures 5-6, Table 6 and the feature-family comparison in Appendix Figure B1.
Figure 5 illustrates the architecture; the evaluation commands produce metrics and correlation plots.

## Workflow

Extract all 24 raw DINOv2 ViT-L/14 block CLS vectors and CLIP image embeddings for NSD images.
Average DINO blocks into early/middle/late groups of eight (three 1,024-dimensional vectors), then combine
them with the 768-dimensional CLIP image vector. A neural encoder and FiLM-conditioned head use a learned
target embedding and CLIP target text to predict a residual mean and log variance. Add the category mean
for full-score prediction.

`--features all`, `clip_only` and `dino_only` select the image feature families. Training provides `tiny`,
`small`, `medium` and `direct` architectures; the main command below uses `small` and zero feature noise.

## Inputs, identities and sources

| Option | Contents |
| --- | --- |
| `--nsd-root` | Algonauts 2023 directory with `subj01` through `subj08` training image filenames. |
| `--dino-dir` | `{subject}_dino_layers.npy`, shaped `(subject_images,24,1024)`. |
| `--clip-image-dir` | `{subject}_clip_img.npy`, shaped `(subject_images,768)`. |
| `--clip-text-file` | NPZ mapping 16 canonical target names to 768-dimensional vectors. |
| `--ranking-file` | `image,score,task` ranking, normally [nsd_m2_ranking.csv](../../complexity/nsd_m2_ranking.csv). |

Use the [Algonauts 2023 challenge images](https://algonautsproject.com/2023/challenge.html#challenge-data).
The [Route B input layout](../../route_b/README.md#inputs-identities-and-sources) shows their filenames.
This variant uses those names to locate feature rows; it does not load fMRI. For each NSD physical image,
features come from the first subject containing it. The loader retains every matched training ranking record,
including repeated physical image-target pairs: 12,108 records across 9,990 physical images for the fixed input.
The 16-category order and text prompts are listed in the [shared README](../../shared/README.md).

DINO uses Torch Hub `facebookresearch/dinov2`, model `dinov2_vitl14`, with its supplied repository default.
CLIP uses [OpenCLIP](https://github.com/mlfoundations/open_clip), model `ViT-L-14` and
[`laion2b_s32b_b82k`](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) weights.
DINO preprocessing is RGB, bicubic resize to 256, center crop 224, and ImageNet normalization.
CLIP uses its pretrained image transform and L2-normalized vectors. Text vectors use five normalized
prompt vectors, their mean, then another L2 normalization.

`extract-features` defaults to batch size 64 for both image extractors and eight workers.
Its extractors substitute a gray image on image-read failure. Keep image order and subject file layouts
fixed. `TORCH_HOME` and `HF_HOME` select model-weight caches; the CLI feature paths select extracted arrays.

## Installation

From the repository root, using Python 3.12:

```bash
python3.12 -m venv .venv-embedding
source .venv-embedding/bin/activate
python -m pip install -c route_a/embedding/requirements.lock.txt '.[embedding]'
```

The [lockfile](requirements.lock.txt) records the bounded software-validation environment.
The default device is CUDA, with CPU fallback in training/evaluation when CUDA is unavailable.
Use an explicit supported device for extraction.

## Commands

The following bash variables locate external inputs and outputs. Run from the repository root, or use an
absolute ranking path after installation:

```bash
NSD_ROOT=/path/to/algonauts_2023_challenge_data
FEATURES=/path/to/embedding-features
RANKING_FILE="$PWD/complexity/nsd_m2_ranking.csv"
OUTPUT_DIR=/path/to/embedding-results

python -m route_a.embedding extract-features --nsd-root "$NSD_ROOT" \
  --dino-dir "$FEATURES/dino" --clip-image-dir "$FEATURES/clip_image" \
  --clip-text-file "$FEATURES/clip_text/clip_text_embeddings.npz"

COMMON=(--nsd-root "$NSD_ROOT" --dino-dir "$FEATURES/dino" --clip-image-dir "$FEATURES/clip_image"
        --clip-text-file "$FEATURES/clip_text/clip_text_embeddings.npz" --ranking-file "$RANKING_FILE")
python -m route_a.embedding train "${COMMON[@]}" --output-dir "$OUTPUT_DIR" \
  --arch small --features all --cv-folds 5 --epochs 100 --noise 0.0

CHECKPOINT_GLOB="$OUTPUT_DIR/checkpoints/route_a_embedding_all_small_noise0.00_fold*of5.pt"
python -m route_a.embedding evaluate-global "${COMMON[@]}" --output-dir "$OUTPUT_DIR" \
  --checkpoint-glob "$CHECKPOINT_GLOB"
python -m route_a.embedding evaluate-per-subject "${COMMON[@]}" --output-dir "$OUTPUT_DIR" \
  --checkpoint-glob "$CHECKPOINT_GLOB"
```

For feature-family experiments, repeat training with `--features clip_only` or `--features dino_only` and
select the matching checkpoint glob during evaluation. Keep deployment-only checkpoints outside that glob.
`route-a-embedding` provides the same subcommands. Feature extraction skips existing subject arrays;
training creates new fold outputs and has no interruption-resume interface.

## Outputs

Training writes `checkpoints/route_a_embedding_{features}_{arch}_noise{noise}_fold{k}of{n}.pt`,
`..._deploy_only.pt`, and `kfold_results_{features}_{arch}_noise{noise}.json` below `--output-dir`.
Fold checkpoints retain validation row indices, model tensors and category means. Deployment training
uses the full dataset and produces a separate checkpoint.

Global evaluation writes `metrics_route_a_embedding_{features}_{arch}.json` and calibration/category plots
in `figures/`. Per-subject evaluation writes `per_subject_metrics_route_a_embedding_{features}_{arch}.json`
and a per-subject correlation plot. Metrics restore category means for full scores and also report residuals.

## Paper targets and reproducibility settings

| Table 6 target | Pearson | Spearman | MAE | R-squared |
| --- | ---: | ---: | ---: | ---: |
| Full score | 0.6226 | 0.5753 | 0.1785 | 0.3650 |
| Category residual | 0.4197 | 0.4040 | - | - |

The dataset computes category means before splitting. Five shuffled row folds use seed 42 and retain
repeated records. The `small` model uses latent width 512, head width 256, category width 16 and dropout 0.5.
Defaults are AdamW with learning rate 0.001 and weight decay 0.05, cosine scheduling, batch size 64,
maximum 100 epochs, early-stopping patience 20 and noise 0.0.

Early stopping records best-loss correlations; fold checkpoints contain the model's final-epoch tensors.
Global evaluation uses those saved tensors and stored validation indices. Per-subject evaluation assigns
image predictions to subjects who viewed the images, so those counts have a different unit from training rows.
Bootstrap and permutation counts default to 5,000 where offered by the evaluation command.
