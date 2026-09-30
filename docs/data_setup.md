# Data setup for engineered Route A

The engineered Route A package needs image files and a CSV of image-target
complexity labels for training. It does not download COCO-Search18 or NSD and
does not calculate M2 labels from behavioral measurements.

## Canonical training CSV

Use UTF-8 CSV with a header and these columns:

| Column | Required | Meaning |
| --- | --- | --- |
| `image_id` | Yes | Stable ID of the physical scene. Reuse it for the same scene across targets and subjects. |
| `image_path` | Yes | Path to a readable image. Relative paths are resolved under `--image-root`. |
| `target` | Yes | One of the 16 canonical search targets below. |
| `score` | Yes | Finite numeric label to predict. For the paper workflow, supply the M2-derived score. |
| `subject` | No | Subject identifier for row identity; it is not a model feature. |

Supply your own images and labels for the training command.

Use a unique `(image_id, target, subject)` key when `subject` is present, or
`(image_id, target)` without it. Several rows may share `image_id` when targets
or subjects differ. These rows stay together during grouped validation. An NSD
scene should use one ID such as `nsd-00013` wherever it appears, even if each
subject has a separate copy of its image file. If the CSV has a `subject`
column, fill it on every row.

The canonical targets are:

```text
bottle       car        chair       clock
cup          fork       keyboard    laptop
microwave    mouse      oven        potted plant
sink         stop sign  toilet      tv
```

Use the canonical spellings in CSV files, especially for `potted plant`,
`stop sign`, and `tv`. The package validates target names and rejects missing
images, invalid scores, ambiguous row keys, and incompatible feature schemas.

## Predicting a batch

A prediction CSV needs `image_path,target`; `image_id` is optional and is
carried through when supplied. For example:

```csv
image_id,image_path,target
scene-001,scene-001.jpg,chair
scene-001,scene-001.jpg,cup
```

Run:

```bash
python -m route_a.engineered predict --model /path/to/model_bundle \
  --input pairs.csv --image-root /path/to/images --out predictions.csv
```

The output must be a different file from the input. If `predictions.csv`
already exists, add `--overwrite` to replace it.

## COCO-Search18 versus NSD

The two adapters in `route_a.engineered.datasets` prepare different source
tables for the same training command. Neither dataset is downloaded.

| | COCO-Search18 | NSD |
| --- | --- | --- |
| Adapter | `load_coco_pairs(csv_path, image_root)` | `load_nsd_pairs(inventory_path, image_root, subject=None)` |
| Source columns | `image,task,score` | `nsd_id,subject,complexity_category,complexity_score,has_complexity` |
| Training label | Supplied `score` | `complexity_score` where `has_complexity` is true |
| Original paper label | M2 log-response-time ranking | M2 fixation-count inventory |
| Subject | Not used | Retained for subject-specific rows; optional adapter filter |
| Targets | Drops `bowl` and `knife`; keeps 16 | Normalizes 16 supported targets; rejects others |

Both adapters return `image_id,image_path,target,score`; the NSD adapter also
returns `subject`. Train a separate model bundle for each dataset and score
definition. The training command does not combine them automatically. The
adapters copy the supplied numeric labels; they do not verify that a source
table contains the paper's M2 scores.

```python
from route_a.engineered.datasets import load_coco_pairs, load_nsd_pairs

# Choose one source for this training run.
labels = load_coco_pairs('/path/to/coco_ranking.csv', '/path/to/coco-images')
# Or: labels = load_nsd_pairs('/path/to/nsd_inventory.csv', '/path/to/nsd-images')
labels.to_csv('labels.csv', index=False)
```

`load_nsd_pairs(..., subject='subj01')` limits the table to one subject.
The NSD adapter uses one physical `nsd-00013` style image ID across subjects
and targets, even when the image files live under separate subject directories.
The COCO adapter uses an embedded NSD token or removes a known target suffix
from the filename to keep one scene under one ID; other files use their stem.
If your source has another layout, prepare the four canonical columns directly.

## Image-model weights and caches

Feature extraction uses these pretrained models:

| Feature family | Weight identifier |
| --- | --- |
| DINOv2 via timm | `vit_small_patch14_dinov2.lvd142m` |
| OpenCLIP | `ViT-L-14` with `laion2b_s32b_b82k` |
| Faster R-CNN via torchvision | `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` |

The first run may download several large weight files. Set library cache paths
before launching Python if you want those files outside your home directory:

```bash
export HF_HOME=/data/model-weights/huggingface
export TORCH_HOME=/data/model-weights/torch
```

`HF_HOME` configures the Hugging Face cache used by Hub downloads;
`TORCH_HOME` configures the PyTorch Hub cache used by torchvision weights.
See the [Hugging Face cache settings](https://huggingface.co/docs/huggingface_hub/main/package_reference/environment_variables)
and [PyTorch Hub cache settings](https://docs.pytorch.org/docs/stable/hub.html).
Populate weight caches before offline use. The trained bundle records weight
identifiers but does not contain the pretrained image-model weights.

The `--cache-dir` option is a separate cache for derived image features. Its
entries depend on the image file and extractor settings, so changing either
creates a new entry. Use an external directory with enough space and reuse it
for later runs:

```bash
python -m route_a.engineered train --data labels.csv --image-root /data/images \
  --out /data/models/route-a --cache-dir /data/cache/route-a
```

`--device auto` chooses an available supported accelerator or CPU. CPU is
supported but feature extraction can take substantial time. For CUDA, install
a PyTorch build appropriate for the host GPU and driver before installing this
package; the package does not require CUDA to install or train. Use `--device
cpu` to request CPU explicitly. Training and prediction use the same feature
extractor configuration recorded in the model bundle.
