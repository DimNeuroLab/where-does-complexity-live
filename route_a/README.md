# Route A: stimulus readout

## Purpose and paper mapping

Route A predicts target-conditioned complexity from an image and the search target.
Both variants learn supplied complexity labels from the [measurement component](../complexity/README.md).
Their trained models use image content and target conditioning at inference.

| Variant | Inputs and model | Paper mapping |
| --- | --- | --- |
| [Engineered](engineered/README.md) | DINO-small, CLIP, detector evidence and target indicators; XGBoost. | Sections 5.5-5.6, engineered stimulus readout. |
| [Embedding](embedding/README.md) | DINO-large and CLIP embeddings; conditional neural predictor. | Section 5.6.1, Figures 5-6, Table 6 and feature-family Figure B1. |

## Workflow

Prepare labels and images, obtain each variant's own features, train and evaluate its predictor, then use
the saved model with the matching preprocessing. The variants have different backbones, validation folds
and saved-model formats. Their READMEs specify those choices and their commands.

## Inputs and installation

The engineered variant accepts canonical image-path/target/score rows, with COCO and NSD adapters.
The embedding variant reads the fixed NSD ranking and cached features indexed through Algonauts image
filenames. A common label source does not imply the same number of training or evaluated observations.

Install the `engineered` or `embedding` extra with the lockfile inside that variant's directory.
Use the corresponding `python -m route_a.engineered` or `python -m route_a.embedding` command.

## Outputs and paper targets

Engineered training writes an XGBoost bundle and grouped validation predictions. Embedding training
writes fold checkpoints, a deployment checkpoint, metrics and evaluation figures. See the variant
READMEs for input schemas, output names and reference values from the manuscript.
