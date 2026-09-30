# Route B reproduction results

The original-protocol Route B rerun completed on September 14, 2026. It included
fresh PCA for all eight subjects, five encoder settings, five head folds per
setting, evaluation, and prediction variance. Recorded compute was 7.27 GPU-hours,
including checkpoint validation. No final-result tuning was performed.

All saved-checkpoint comparisons produced identical observation identities,
predictions, log variances, and targets in original and ported code, with maximum
absolute difference zero. This reproduces saved inference. Fresh training used
recorded settings and recovered defaults where historical launch metadata was absent.

| Visual weight | Paper Pearson | Fresh Pearson | Paper Spearman | Fresh Spearman | Paper MAE | Fresh MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.499404 | 0.500561 | 0.434136 | 0.436335 | 0.193179 | 0.193056 |
| 0.25 | 0.593909 | 0.614741 | 0.549363 | 0.571343 | 0.178447 | 0.174596 |
| 0.5 | 0.610637 | 0.623225 | 0.569871 | 0.580794 | 0.175412 | 0.172926 |
| 0.75 | 0.608604 | 0.608443 | 0.562320 | 0.565504 | 0.175624 | 0.175764 |
| 1, baseline | 0.612456 | 0.607412 | 0.567506 | 0.562107 | 0.175223 | 0.176004 |

The fresh baseline is slightly below the paper reference. Two sweep configurations
exceed its Pearson correlation. These values are descriptive results of the
original protocol, not estimates under the later corrected validation protocol.

For the 1,173 images with at least two predicted target scores, the paper checkpoint
has total variance 0.025626, within-image variance 0.007898 (30.82%), and between-image
variance 0.017728. Fresh baseline values are 0.025774, 0.007615 (29.55%), and 0.018159.

The [machine-readable report](../route_b/reference/reproduction.json) contains
pooled, per-subject, and per-category metrics, equivalence checks, and exact settings.
[Figure 9 comparison PDF](../route_b/reference/comparison.pdf) and
[PNG](../route_b/reference/comparison.png) are available for inspection.

## Interpretation and manuscript alignment

The release preserves shared PCA and pretrained encoders across head folds,
stage-dependent normalization, validation-based checkpoint selection, active
encoder dropout during frozen-head training, repeated ranking records, and the
single saved category-mean dictionary used across evaluation folds. These choices
are explained in the [inventory](route_b_inventory.md).

Paper wording about strictly training-only preprocessing, encoder exclusion of all
head-validation images, and fold-specific residual means does not describe every
recovered implementation detail. Manuscript alignment remains an author task.
This release reports the implemented protocol and its results explicitly; cleanup
and packaging do not resolve those scientific protocol differences.

## Public workflow validation

A new Python 3.12 venv was installed from the pinned requirements using pip; its
dependency check passed. The model weights were downloaded into empty dedicated
caches. Validation used a Route B-only source export with the ranking supplied as
an external input, without loading the legacy repository in public commands.

Fourteen scientific tests passed, covering original equivalence, interrupted
training recovery, configuration, input protection, and feature-free evaluation.
A public training run completed one encoder epoch and one epoch in each of five
head folds, followed by evaluation. This smoke run reused verified full feature
and PCA arrays as external data; model extraction was checked separately.

The public saved-model workflow regenerated text vectors, transformed downloaded
fMRI using the saved PCA parameters, and evaluated all 18,736 observations.
Every prediction matched the paper reference within fixed `rtol=1e-6`, `atol=1e-6`;
maximum prediction difference was `1.4901161193847656e-7`. The CPU prediction CLI
also passed on four aligned rows, with maximum difference `4.842877388000488e-8`.

The DINOv2 extractor matched the original source and saved features exactly on 64
images at batch size 64. CLIP image features matched the saved bank within
`1.1920928955078125e-7` on 128 images at batch size 128; all target vectors matched
exactly. These checks use the public model sources and pinned revisions.

The [validation evidence](../route_b/reference/publication_validation.json) retains
the checks and their scope. The full feature banks and full training experiments
were not regenerated again during packaging. Computational training logic is
unchanged, and the earlier full rerun remains the performance record.

A completed public evaluation was resumed successfully: all five fold files were
reused without changing their hashes or timestamps, and GPU accounting remained
unchanged. Standalone test discovery passes nine tests and skips five optional
original-source comparisons; enabling the original source passes all fourteen.

The temporary model package used during validation was removed from release
preparation. Public inference uses locally trained checkpoints and PCA outputs;
only compact results, figures, and validation records are retained for release.
