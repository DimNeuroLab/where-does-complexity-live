# Route A: engineered stimulus readout

## Purpose and paper mapping

This variant predicts an image-target complexity score from DINOv2 and CLIP image embeddings, Faster R-CNN
detection evidence and target indicators with XGBoost. It implements the engineered stimulus-readout
workflow associated with Sections 5.5-5.6. Training learns the scale of the supplied labels.

## Workflow

1. Validate labeled image-target rows and physical-image identities.
2. Extract DINO-small (384 dimensions), CLIP (768), 26 detector features and 16 target indicators.
3. Fit median imputation and separate DINO/CLIP PCA transforms on each training fold, then fit XGBoost.
4. Export grouped held-out predictions and fit a final full-data model bundle for prediction.

DINO and CLIP each use up to 64 principal components, limited by the available training rows.
Grouped validation keeps every row for a physical image in one fold. It uses shuffled
`StratifiedGroupKFold` when per-target image counts permit, otherwise `GroupKFold`, with at most the
requested fold count. Final prediction uses the saved preprocessing and extractor configuration.

## Inputs, identities and sources

Training accepts a UTF-8 CSV with these columns:

| Column | Meaning |
| --- | --- |
| `image_id` | Stable physical-scene ID, shared across targets and subject copies. |
| `image_path` | Readable image path; relative paths resolve under `--image-root`, or the launch directory if omitted. |
| `target` | Canonical target name. |
| `score` | Finite numeric complexity label. |
| `subject` | Optional subject identity, filled on every row when present; not a model feature. |

Keys must be unique: `(image_id,target)` or `(subject,image_id,target)` when subject is present.
Prediction CSVs require `image_path,target`; optional identity columns are carried through.

The 16 supported targets, in feature order, are:

```text
bottle, car, chair, clock, cup, fork, keyboard, laptop,
microwave, mouse, oven, potted plant, sink, stop sign, toilet, tv
```

Two Python adapters prepare the canonical schema:

| Adapter in [datasets.py](datasets.py) | Source schema and selection |
| --- | --- |
| `load_coco_pairs(csv_path, image_root)` | `image,task,score`; drops bowl and knife and normalizes target names. |
| `load_nsd_pairs(inventory_path, image_root, subject=None)` | `nsd_id,subject,complexity_category,complexity_score,has_complexity`; selects labeled rows and retains subject identity. |

COCO labels use the supplied M2 RT ranking. NSD inventory labels use supplied M2 count scores.
Prepare and train each label source separately. The adapters copy label values; the training command
accepts the canonical schema and does not generate labels itself.

```python
from route_a.engineered.datasets import load_coco_pairs

labels = load_coco_pairs('/path/to/coco_ranking.csv', '/path/to/coco/images')
labels.to_csv('/path/to/labels.csv', index=False)
```

Obtain [COCO-Search18](https://sites.google.com/view/cocosearch/) or
[Algonauts 2023](https://algonautsproject.com/2023/challenge.html#challenge-data) images externally.
NSD physical IDs use the same `nsd-00013` identity across subjects; the adapter resolves the corresponding
image files. The source inventory is a different schema from the released ranking CSV.

| Feature | Weight identifier and source |
| --- | --- |
| DINOv2 | [`vit_small_patch14_dinov2.lvd142m`](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m), loaded with timm. |
| CLIP | [`ViT-L-14`, `laion2b_s32b_b82k`](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K), loaded with OpenCLIP. |
| Detector | [`FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html). |

DINO uses its timm pretrained transform; CLIP uses its pretrained transform and L2 normalization.
Detector evidence uses the top 50 boxes and confidence thresholds 0.5 and 0.7; geometry and target/distractor
summaries are defined in [features.py](features.py). Weights use the libraries' model caches. Set `HF_HOME`
and `TORCH_HOME` before Python starts to locate those caches externally. `--features-dir` instead selects
the derived-feature cache; entries depend on image file metadata and extractor settings.

## Installation

From the repository root, using Python 3.12:

```bash
python3.12 -m venv .venv-engineered
source .venv-engineered/bin/activate
python -m pip install -c route_a/engineered/requirements.lock.txt '.[engineered,dev]'
```

The [lockfile](requirements.lock.txt) records the bounded software-validation environment.
The feature extractor supports `--device auto`, `cpu` and `cuda[:index]`; the regressor uses its configured
CPU threads. Populate weight caches before offline feature extraction.

## Commands

```bash
python -m route_a.engineered train --labels-file /path/to/labels.csv \
  --image-root /path/to/images --output-dir /path/to/model-bundle --features-dir /path/to/features
python -m route_a.engineered predict --model-dir /path/to/model-bundle \
  --image-file /path/to/image.jpg --target chair
python -m route_a.engineered predict --model-dir /path/to/model-bundle \
  --input-file /path/to/pairs.csv --image-root /path/to/images --output-file /path/to/predictions.csv
```

`route-a-engineered` is the equivalent console command. Training accepts `--n-estimators`,
`--pca-components`, `--cv-folds` and `--seed`. Use `--cv-folds 0` to skip validation.
Use `--overwrite` when intentionally replacing an existing bundle or prediction CSV. Batch output must
differ from its input CSV. Prediction can override `--device` and reuse `--features-dir`.

The Python interface is `EngineeredPredictor.from_bundle(...).predict_image(image_path, target)` or
`predict_batch(frame, image_root=...)`; the feature-cache keyword in Python is `cache_dir`.

## Outputs

Keep `model.json`, `preprocessing.joblib` and `metadata.json` together as one bundle. It stores the regressor,
preprocessing, feature schema, weight identifiers and training configuration. Pretrained image-model weights
stay in their separate caches. Grouped training also writes `validation_predictions.csv` with identities,
labels, predictions and fold numbers. Single-image prediction prints JSON; batch prediction writes CSV.

## Paper targets and reproducibility settings

The manuscript's Table 4 gives MAE 0.167, Spearman 0.650, Pearson 0.677 and Kendall 0.469 for its
“All together PCA-64” row, on 12,476 samples. Those values are paper targets with the manuscript's population
unit; the command's training population is the supplied canonical table and is recorded in bundle metadata.

The submitted command defaults are seed 42, five grouped folds, separate PCA-64 features, 300 trees,
absolute-error objective, learning rate 0.03, maximum depth 6, row and column subsampling 0.8, L2 penalty 1.0,
`hist` tree method and four CPU jobs. [training.py](training.py) and [preprocessing.py](preprocessing.py)
define the full recipe and saved transforms.
