# Shared embedding helpers

## Purpose and workflow

These helpers support [embedding Route A](../route_a/embedding/README.md): image/target embeddings,
complexity-record loading, target conditioning and correlation metrics. The engineered Route A and Route B
workflows retain their own scientific implementations and defaults.

| Module | Responsibility |
| --- | --- |
| [constants.py](constants.py) | Subject/category order, embedding dimensions, DINO tier definitions and seed. |
| [nsd_utils.py](nsd_utils.py) | Parse NSD filenames and list subject training images. |
| [complexity_data.py](complexity_data.py) | Read ranking records and cached CLIP target embeddings. |
| [features.py](features.py) | Extract DINO block CLS and CLIP image/text features. |
| [complexity_head.py](complexity_head.py) | FiLM target-conditioned mean/log-variance head and Gaussian loss. |
| [metrics.py](metrics.py) | Correlations, bootstrap, permutation tests and significance utilities. |

## Inputs, identities and sources

Subjects are `subj01` through `subj08`. The category order is:

```text
bottle, car, chair, clock, cup, fork, keyboard, laptop,
microwave, mouse, oven, potted plant, sink, stop sign, toilet, tv
```

Ranking columns are `image,score,task`. Image labels encode split, one-based subject image index, physical
NSD ID and an optional target suffix. `load_complexity_records` defaults to training rows and keeps repeats.
Cached features follow the sorted Algonauts training-image rows; text NPZ keys are the canonical category names.
See the consumer's [input and model details](../route_a/embedding/README.md#inputs-identities-and-sources).

CLIP target prompt templates are:

```text
a photo of a {}.
a photograph of a {}.
the target object is a {}.
a {} in a scene.
a photo containing a {}.
```

Each prompt vector is normalized, averaged across templates, then normalized again. DINO features are raw
block CLS vectors; consecutive groups of eight of the 24 blocks define the early/mid/late tiers.

## Installation and commands

Install the `embedding` extra with [its lockfile](../route_a/embedding/requirements.lock.txt).
These modules are imported by the embedding workflow; the public entry point is
`python -m route_a.embedding`, including its `extract-features` subcommand.

## Outputs and reproducibility settings

Extraction writes per-subject NumPy arrays and a CLIP text NPZ. The default extraction batch sizes at
helper level are 64 for DINO and 128 for CLIP; the embedding CLI explicitly passes its batch-size setting,
64 by default, to both. Keep that distinction when invoking helpers directly.

The shared head's default dropout is 0.2. Embedding Route A passes its architecture's own dropout instead.
The model constructor, not a shared default, defines the selected scientific recipe. See the consumer's
[paper targets and settings](../route_a/embedding/README.md#paper-targets-and-reproducibility-settings).
