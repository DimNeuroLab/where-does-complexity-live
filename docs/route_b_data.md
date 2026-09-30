# Route B data and pretrained models

Route B uses public images and brain responses as external inputs. This repository
provides extraction and training code, instructions, and compact scientific records.
Images, fMRI responses, and extracted image or text embeddings are not redistributed.

## Datasets

| Resource | Access | Use in Route B |
| --- | --- | --- |
| Natural Scenes Dataset (NSD) | [Project and data access](https://www.naturalscenesdataset.org/) | Original source of the eight subjects' brain responses and image identities. Follow the project's data access agreement. |
| Algonauts Project 2023 | [Challenge data download and format](https://algonautsproject.com/2023/challenge.html#challenge-data), [official tutorial](https://github.com/gifale95/algonauts_2023) | The specific packaged images, hemisphere arrays, and stream ROI masks consumed by this code. |
| MS-COCO | [Dataset](https://cocodataset.org/#download) | Source images underlying NSD. Use the challenge's cropped PNGs for Route B extraction. |
| Complexity labels | [Preserved ranking](../complexity/nsd_m2_ranking.csv) | Fixed image-target supervision supplied by the separate complexity component. |

Downloading raw NSD alone does not create the expected Algonauts directory layout.
Use the 2023 challenge package. Its hemisphere arrays are already standardized
within scan sessions and averaged across repeated presentations. Do not replace
them with an arbitrary raw NSD beta release or average them again.

```text
${NSD_DATA}/
  subj01/                         # Repeat through subj08
    training_split/
      training_images/
        train-0001_nsd-00013.png
        ...
      training_fmri/
        lh_training_fmri.npy
        rh_training_fmri.npy
    roi_masks/
      lh.streams_challenge_space.npy
      rh.streams_challenge_space.npy
```

The first image index is one-based and identifies its fMRI row; the NSD ID is
zero-based and identifies the physical image across subjects. Keep the original
filenames and row order. Subject image counts are 9841, 9841, 9082, 8779, 9841,
9082, 9841, and 8779. The stream arrays must match the corresponding hemisphere
vertex order. Route B uses training-split data, with its own five image-grouped
folds; the challenge's unreleased test fMRI is not needed.

## Model sources and extraction

| Representation | Exact model | Extraction code |
| --- | --- | --- |
| DINOv2 visual targets | [Meta DINOv2 ViT-L/14 without registers](https://github.com/facebookresearch/dinov2), hub name `dinov2_vitl14`, [weights](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) | [features/dino.py](../route_b/features/dino.py) |
| CLIP image and target vectors | [LAION CLIP ViT-L/14](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K), OpenCLIP model `ViT-L-14`, pretrained tag `laion2b_s32b_b82k` | [features/clip.py](../route_b/features/clip.py), using [OpenCLIP](https://github.com/mlfoundations/open_clip) |

DINOv2 source is pinned to
[`7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c`](https://github.com/facebookresearch/dinov2/tree/7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c).
The recorded CLIP model repository revision is
[`1627032197142fbe2a7cfec626f4ced3ae60d07a`](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/tree/1627032197142fbe2a7cfec626f4ced3ae60d07a).
The exact tested Python dependencies are pinned in
[requirements.lock.txt](../route_b/requirements.lock.txt), including OpenCLIP 3.2.0.

DINOv2 uses RGB images, bicubic resize of the shorter edge to 256, a 224-pixel
center crop, and ImageNet normalization. The extractor retains the CLS token
immediately after each of the 24 blocks, before final backbone normalization.
Training averages blocks into three consecutive groups of eight, each with
1024 features. Using the model's final normalized output changes the experiment.

CLIP uses its pretrained image transform, L2-normalized 768-dimensional image
vectors, and five target prompts. Text vectors are normalized per prompt,
averaged, and normalized again. All prompts and the 16-category order are in the
source. Text vectors are regenerated locally even for inference-only use.
Image embeddings are training supervision and are unnecessary for prediction.

## Download once, extract locally

From the repository root, after installing the Route B venv:

```bash
export NSD_DATA=/path/to/algonauts_2023_challenge_data
export ROUTE_B_OUTPUT=/path/to/new/route_b_run
export DINOV2_ROOT=/path/to/dinov2

git clone https://github.com/facebookresearch/dinov2.git "$DINOV2_ROOT"
git -C "$DINOV2_ROOT" checkout 7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c
cp route_b/experiments/paper.example.json /path/to/route_b.json
.venv-routeb/bin/python -m route_b.run --config /path/to/route_b.json --stages features
```

The first extraction downloads the public model weights. Network access is only
needed for obtaining data, source, and pretrained weights. Set `TORCH_HOME` and
`HF_HOME` to explicit directories when storing caches outside the home directory.
The public runner does not force offline mode. Use the pinned model download helper
below to populate these caches and verify the recorded CLIP revision:

```bash
.venv-routeb/bin/python -m route_b.features.download_models
```

To generate only target embeddings, use `--stages text`. Subsequent commands in the
same output directory must include `--resume`. Feature arrays are written under
`features/dino`, `features/clip_image`, and `features/clip_text` within the run.
Existing feature arrays can be supplied with `inputs.features` in a private config;
this is optional and does not require a legacy repository.

The original extractors replace unreadable images with a gray image. Confirm the
public challenge files are complete before extracting; changing missing-image
handling changes the original experiment. Dataset and upstream model terms remain
those of their providers; this repository does not rehost their data or model files.

Use the original extraction batch sizes for reproduction: 64 for DINOv2 and 128
for CLIP images. A validation batch of four produced different features because
GPU execution depends on batch shape. At the original sizes, 64 DINOv2 images
matched the historical bank exactly and 128 CLIP images matched within `1e-6`.
All 16 CLIP target vectors matched exactly. The model files' recorded SHA-256
hashes are in [upstream_models.json](../route_b/reference/upstream_models.json).
