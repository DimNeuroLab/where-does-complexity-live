# Route B installation and reproduction

Route B runs independently of the original `preds` workspace. Obtain the public
inputs and model sources using the [data guide](route_b_data.md). This release
preserves the original protocol, including the differences recorded in the
[inventory](route_b_inventory.md). It does not implement the later corrected protocol.

## Installation

Use Python 3.12 on Linux. The tested stack is PyTorch 2.10.0 with CUDA 12.8,
OpenCLIP 3.2.0, NumPy 2.4.2, and scikit-learn 1.8.0. Install in a separate venv:

```bash
python3.12 -m venv .venv-routeb
.venv-routeb/bin/python -m pip install -r route_b/requirements.lock.txt
.venv-routeb/bin/python -m pip check
```

Run commands from the repository root. Training and full paper evaluation require
a CUDA GPU. The separate prediction command also supports CPU. The original
DataParallel calls compute on the first configured GPU; `[0]` is the public default.
Use roughly 16 GB or more of GPU memory, up to 128 GB of host RAM, and substantial
external disk space for features and checkpoints. Actual capacity depends on batch
size and retained experiments; the tested full run used 48 GB GPUs.

The example settings use batch size 256, encoder maximum 100 epochs, head maximum
60 epochs, patience 15, learning rate 0.0003, AdamW weight decay 0.01, a frozen
encoder, and zero auxiliary head loss. The head settings were recovered from a
checkpoint tensor-identical to the paper baseline. Missing historical encoder
launch metadata was filled from source defaults. Seeds and folds remain fixed.

## Public pipeline

Set the three environment variables in the data guide, then edit a copy of
[paper.example.json](../route_b/experiments/paper.example.json). No original
repository, historical PCA, or saved checkpoints are required for a fresh run.

```bash
.venv-routeb/bin/python -m route_b.run --config /path/to/route_b.json
```

Default order: extract visual/text features, fit per-subject PCA, pretrain the
baseline encoder, train five heads, evaluate, repeat the four additional visual
weights for Figure 9, then compute prediction variance. Features produced by the
first stage are consumed by later stages automatically.

Stages can also be invoked separately:

```bash
.venv-routeb/bin/python -m route_b.run --config /path/to/route_b.json --stages features prepare
.venv-routeb/bin/python -m route_b.run --config /path/to/route_b.json --stages pretrain train evaluate --resume
.venv-routeb/bin/python -m route_b.run --config /path/to/route_b.json --stages sweep variance --resume
```

`--weight` selects 0, 0.25, 0.5, 0.75, or 1 for an individual encoder/head/evaluation.
`sweep` runs the four nonbaseline weights. Short validation runs can use a separate
config with one encoder epoch and one head epoch. Such runs test executability,
not scientific performance, and must use separate output directories.

Each run records source hashes, resolved configuration, dependencies, input
fingerprints, and completed artifacts. Resume rejects changed inputs, source,
settings, or dependencies. Training stores model, optimizer, scheduler, early
stopping, selected weights, and random states after each epoch. A partial epoch
repeats after interruption. Completed stages and PCA subjects are reused.

## Persistent execution

A user systemd service can run the public command with an absolute Python path,
absolute config path, and the repository root as `WorkingDirectory`. Set
`Restart=on-failure`, `RestartSec=60`, and `RestartPreventExitStatus=75`; use
`KillMode=control-group` and enable user lingering if it must survive logout.
The portable [service template](../route_b/experiments/route_b.service.example)
contains these settings. Substitute paths, install it under
`~/.config/systemd/user/route-b.service`, and run:

```bash
systemctl --user daemon-reload
systemctl --user enable --now route-b.service
journalctl --user -u route-b.service -f
```

The configured GPU-hour allowance includes GPU stages and survives restarts.
CPU PCA is excluded. In-flight operations can slightly exceed the allowance;
exit 75 pauses the job without an automatic restart. `status.json`, `logs/`, and
`recovery/` contain progress. Stop/start pauses/resumes; `disable --now` also
prevents automatic startup after reboot.

## Saved-model evaluation and prediction

Checkpoints and PCA parameters are created by your local training run. The public
release contains [compact results](route_b_artifacts.md), not trained model weights.
Evaluate your completed run with:

```bash
.venv-routeb/bin/python -m route_b.run --config /path/to/route_b.json --stages evaluate variance --resume
```

Full out-of-fold evaluation keeps the ranking's repeated records and reconstructs
five image-grouped folds. It requires all eight subjects. To evaluate existing local
artifacts in a new output directory, set `inputs.pca_models` and `inputs.checkpoints`
to their directories, then run `--stages text transform evaluate variance`.
The transform stage applies your saved PCA models without refitting.

For arbitrary aligned rows from a supported subject, use your locally trained
checkpoint and preprocessing directory:

```bash
.venv-routeb/bin/python -m route_b.predict \
  --checkpoint "$ROUTE_B_OUTPUT/checkpoints/complexity/visual_1.pt" \
  --pca "$ROUTE_B_OUTPUT/pca" --nsd "$NSD_DATA" --subject subj01 \
  --lh /path/to/lh_rows.npy --rh /path/to/rh_rows.npy \
  --tasks /path/to/targets.json --clip-text "$ROUTE_B_OUTPUT/features/clip_text/clip_text_embeddings.npz" \
  --fold 1 --device cuda:0 --output /path/to/predictions.csv
```

The PCA directory contains `subj*_pca_models.pkl` and `subj*_pca_fmri.npy`, generated
locally by preprocessing. Prediction recovers the original normalization statistics
from these local training rows. Neither file type belongs in the public release.

`targets.json` is a JSON list containing one category string per row. Hemisphere
arrays must use the challenge vertex order. Outputs contain row, subject, target,
fold, full complexity, and predicted sigma. These models have subject-specific
adapters for the original eight subjects. For paper observations, use the assigned
held-out fold; arbitrary fold selection is not a held-out accuracy estimate.

## Results and developer comparisons

The full original-protocol rerun is complete. All five saved-checkpoint comparisons
had zero differences in predictions and recovered the reported metrics. Fresh
training is a separate result, summarized in [the release report](route_b_results.md)
and [machine-readable metrics](../route_b/reference/reproduction.json).

Normal usage does not import or require the original source. Optional original
comparisons are available through `--verify-original` and
[the developer config](../route_b/development/replay.example.json). The historical
run record is retained in [development/historical_run.md](../route_b/development/historical_run.md).
Scientific regression tests run with:

```bash
.venv-routeb/bin/python -m unittest discover -s route_b/tests -v
```

Setting `ROUTE_B_ORIGINAL=/path/to/preds` enables the extra original-source tests.
There is no formatter, linter, or style-enforcement dependency.
