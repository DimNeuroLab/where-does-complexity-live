# Route A: engineered image features

This package predicts a complexity score for an **image and search target** using
DINOv2 and OpenCLIP image embeddings, Faster R-CNN detection evidence, target
indicators, and an XGBoost regressor. It provides ordinary image-path training
and inference without requiring NSD or fMRI data.

## Install, train, predict

Use Python 3.10 or newer. From the repository root, create a virtual
environment and install the engineered component:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[engineered]'
```

Train a model from a CSV of labeled image-target pairs:

```bash
python -m route_a.engineered train --data labels.csv --image-root /path/to/images --out model_bundle/
```

Predict the score for a new pair:

```bash
python -m route_a.engineered predict --model model_bundle/ --image /path/to/example.jpg --target chair
```

No complete pretrained model bundle has been published yet. Without a bundle,
train one before prediction. `score` is supplied by the user at training time;
the package does not derive it from response times or eye movements. With the
paper's labels, predictions estimate the M2-derived complexity score. With other
labels, the output follows that label scale.

## COCO-Search18 and NSD

Both can supply training rows, but they are **separate datasets and label
sources**. The COCO adapter reads its supplied `score`; the NSD adapter reads
`complexity_score` from labeled inventory rows and retains `subject`. Each
adapter produces the same four required training columns. Train a separate
bundle for each dataset; the code does not combine them or ship a trained model.
See [data setup](../../docs/data_setup.md#coco-search18-versus-nsd) for the
source columns and distinct paper label definitions. NSD `subject` identifies
rows but does not enter the regressor. Prediction uses an image and target with
either bundle; its score scale comes from that bundle's training labels.

## Input and batch prediction

The training CSV requires `image_id,image_path,target,score`, with optional
`subject`. `image_id` identifies the physical scene across targets and subjects
so validation can keep all rows for that scene in one fold. Relative image paths
are resolved under `--image-root`. See [data setup](../../docs/data_setup.md)
for the full schema.

Batch prediction takes a CSV with `image_path,target` and optional `image_id`:

```bash
python -m route_a.engineered predict --model model_bundle/ --input pairs.csv --image-root /path/to/images --out predictions.csv
```

Choose a new output path for each batch. To replace an existing output CSV,
add `--overwrite`; the input and output CSV cannot be the same file.

Single-pair prediction prints JSON with a `prediction` number. Batch prediction
writes one CSV row per input row with a `prediction` column and available
identity columns. The same interface is available in Python:

```python
import pandas as pd

from route_a.engineered.predictor import EngineeredPredictor

predictor = EngineeredPredictor.from_bundle('model_bundle/', cache_dir='/path/to/feature-cache')
score = predictor.predict_image('/path/to/example.jpg', 'chair')
rows = pd.read_csv('pairs.csv')
predictions = predictor.predict_batch(rows, image_root='/path/to/images')
```

The 16 canonical targets are `bottle`, `car`, `chair`, `clock`, `cup`, `fork`,
`keyboard`, `laptop`, `microwave`, `mouse`, `oven`, `potted plant`, `sink`,
`stop sign`, `toilet`, and `tv`. Use these spellings in shared data files.

## Runtime options

`--device auto` uses an available supported accelerator and otherwise runs on
CPU. Use `--device cpu` to force CPU execution. The first feature extraction
may download DINOv2, OpenCLIP, and Faster R-CNN weights through their respective
libraries; later runs reuse their local weight caches. `--cache-dir` stores
derived image features for reuse between training or prediction runs. Keep this
cache and the model bundle outside Git. Weight and cache setup is described in
[data setup](../../docs/data_setup.md).

Training also accepts `--n-estimators`, `--pca-components`, and `--cv-folds`.
The configuration and learned preprocessing are saved in the bundle. The default
validation groups rows by physical `image_id` and fits transforms on training
folds only. Training writes `validation_predictions.csv` next to the three
bundle files when validation is enabled. Use `--cv-folds 0` to skip validation;
use `--overwrite` to replace files in an existing output directory.

Keep `model.json`, `preprocessing.joblib`, and `metadata.json` together when
moving a trained bundle. The image-model weights are downloaded separately and
are not stored in these files.

## Scope

This package covers the engineered image-and-target Route A predictor. The
embedding-conditioned neural predictor and paper result reproduction are outside
this first release.
