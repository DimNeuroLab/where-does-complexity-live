# Route B release contents

The repository contains code, reproduction instructions, public dataset/model
references, and compact scientific results. Trained Route B checkpoints, fitted
PCA parameters, datasets, and extracted embeddings are generated or obtained
locally and are not distributed with the release.

## Retained scientific records

| File | Contents |
| --- | --- |
| [paper_metrics.csv](../route_b/reference/paper_metrics.csv) | Pooled paper-reference metrics for the baseline and Figure 9 settings |
| [paper_subject_metrics.csv](../route_b/reference/paper_subject_metrics.csv) | Paper-reference metrics by subject and visual weight |
| [paper_variance.csv](../route_b/reference/paper_variance.csv) | Paper-reference prediction variance decomposition |
| [reproduction.json](../route_b/reference/reproduction.json) | Separately labeled paper-reference and fresh-training results, including equivalence checks |
| [comparison.md](../route_b/reference/comparison.md) | Human-readable reproduction comparison |
| [comparison.pdf](../route_b/reference/comparison.pdf), [PNG](../route_b/reference/comparison.png) | Reproduction comparison plots, distinguished from the original manuscript figures |
| [publication_validation.json](../route_b/reference/publication_validation.json) | Public workflow validation record |
| [upstream_models.json](../route_b/reference/upstream_models.json) | Public pretrained model identities and SHA-256 hashes, without weights |

The existing complexity ranking remains a separate component's input to Route B.
This release does not alter that component.

## Generate models locally

Obtain the external inputs using the [data/model guide](route_b_data.md), then
follow the [training and prediction instructions](route_b_reproduction.md).
Training writes checkpoints, PCA artifacts, and features into the configured
output directory. Use these local outputs for subsequent evaluation and inference.

Exact replay of the historical paper checkpoints is documented as a completed
validation experiment. Those checkpoints are not included in the public release;
a fresh training run can differ numerically, as recorded in the results report.
