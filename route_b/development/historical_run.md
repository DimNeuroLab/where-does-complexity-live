# Original Route B reproduction

The port preserves the original implementation used to produce the reported results.
Its source mapping is in `route_b/experiments/source_mapping.json`. The existing
[inventory and audit](../../docs/route_b_inventory.md) remain the provenance record. This is
separate from the corrected validation and optimization experiments in `preds`.

## Numerical behavior

The port retains the original shared PCA and encoder, fixed encoder holdout,
stage-specific normalization, record ordering and duplication, sampler, dropout,
validation-based early stopping, and single saved category-mean dictionary.
The code does not silently substitute fold-specific preprocessing or category means.
Checkpoint parameter names and dimensions are unchanged.

The fresh recipe uses 100 encoder epochs, 60 head epochs, batch size 256, patience 15,
learning rate 0.0003, AdamW weight decay 0.01, and the original seed/fold logic.
The head keeps the encoder frozen and uses auxiliary weight zero. The named
`routeb_pretrained_aux0_frozen.pt` checkpoint records these head settings; every
tensor in all five folds and its category means exactly match `multisubj_best.pt`.
The original encoder has incomplete metadata, so unrecorded settings use recovered
source defaults. This is not a claim that every historical launch argument is known.

The four extra Figure 9 points use visual weights 0, 0.25, 0.5, and 0.75, with
complementary reconstruction weights. The fresh baseline provides weight 1.
No tuning is performed against the final evaluation results.

## Environment and data

The dedicated environment was created with `python3.12 -m venv` at
`/home/riccardo/data/routeb-original-port-2026-09-12/venv`. Installed packages were
copied into that environment from the existing `preds` venv without network downloads.
The package lists match, and `pip check` passes. The separate copy keeps dependencies
off the nearly full home volume and does not share a site-packages symlink.
The local `route_b/.venv` link points to this environment.

Each run saves Python, PyTorch, NumPy, and the complete package list in its manifest
and `requirements.freeze.txt`. To rebuild on a compatible CUDA machine:

```bash
python3.12 -m venv route_b/.venv
route_b/.venv/bin/python -m pip install -r route_b/requirements.lock.txt
```

The reproduction reuses the verified original DINOv2 and CLIP image/text features
and the preserved paper ranking. It recomputes PCA from raw fMRI before retraining.
Original data, checkpoints, and previous experiment outputs remain unchanged.

The optional feature extraction stage writes under the new output directory. It
does not replace the configured reference features. A later configuration can use
those exports as its inputs. Feature validation uses locally cached DINOv2 and CLIP
backbone weights; the DINOv2 source checkout and model caches must be available.

## Initialization and execution

Copy `route_b/experiments/paper.example.json`, then fill in the input and output paths.
Keep new outputs separate from preserved inputs. The actual machine configuration
lives in `/home/riccardo/data/routeb-original-port-2026-09-12/config.json`.

```bash
route_b/.venv/bin/python -m route_b.run --config /path/to/config.json --initialize-only
```

Initialization freezes original and ported source, configuration, input hashes,
fold identities, subject row mappings, and the environment. Run the snapshot from
`<output>/source_snapshot/publication` using the dedicated venv:

```bash
/absolute/path/to/venv/bin/python -m route_b.run --config /path/to/output/config.json --resume
```

The default stages are `prepare`, `pretrain`, `train`, `evaluate`, `sweep`, and
`variance`. `--stages` can select a subset, or include `features`; earlier dependencies
must exist when running later stages alone. All invocations first complete the replay
checks, reusing their completed artifacts on resume.

The runner performs, in order:

1. Original-versus-ported DINOv2 and CLIP extraction on four fixed real images and all targets.
2. Original-versus-ported replay of the baseline and all four additional Figure 9 checkpoints.
3. Original-versus-ported prediction variance and the ground-truth population check.
4. Fresh PCA for eight subjects, baseline training/evaluation, four additional sweep runs, and variance analysis.
5. Comparison tables and standalone PDF/PNG figures.

## Validation

```bash
ROUTE_B_ORIGINAL=/path/to/preds route_b/.venv/bin/python -m unittest route_b.tests.test_equivalence -v
```

Scientific tests cover dataset sample order and normalization, PCA, loss values and
gradients, short original-versus-ported training, and encoder/head recovery. Recovery
tests require identical final weights and global random state after interruption.

Full replay requires exact observation identities and counts, including the baseline's
18,736 subject-record observations. Predictions, log variances, and core metrics use
fixed tolerances `rtol=1e-6`, `atol=1e-6`. Prediction variance retains 1,173 images;
the separate ground-truth analysis retains 1,205. A failed comparison stops execution
before fresh training. Tolerances are not widened after observing results.

Replay establishes inference equivalence. Fresh training is a separate reproducibility
experiment under recorded settings. All fresh results and discrepancies are reported;
matching saved inference does not establish exact historical training reproduction.

## Persistent operation

`ijcv-routeb-original-port.service` runs a recorded source snapshot with user lingering
and boot activation. Closing a terminal, SSH, or the conversation does not stop it.
Working-tree edits do not alter a running or resumed experiment.

```bash
systemctl --user status ijcv-routeb-original-port.service
journalctl --user -u ijcv-routeb-original-port.service -f
cat /home/riccardo/data/routeb-original-port-2026-09-12/status.json
```

Detailed task logs are under `<output>/logs/`. Epoch progress and recovery checkpoints
are under `<output>/recovery/`. Stop and resume with:

```bash
systemctl --user stop ijcv-routeb-original-port.service
systemctl --user start ijcv-routeb-original-port.service
```

An intentional stop stays paused until start or reboot. Use `disable --now` to keep it
paused across reboots, and `enable --now` to restore boot activation. Unexpected
failures restart after 60 seconds, with at most three starts in an hour. Resolve
the logged issue before using `reset-failed` and starting again.

The single 72 GPU-hour cap includes validation, baseline, sweep, and prediction
variance. Both configured GPUs are charged conservatively while reserved, even though
the preserved training loops compute on the first one. CPU-only PCA is excluded.
Budget exhaustion exits with code 75 and does not automatically restart or extend
the allowance. A GPU operation already in flight may finish slightly beyond the cap.

Model, optimizer, scheduler, early stopping, selected weights, and Python/NumPy/PyTorch/GPU
random states are saved atomically each epoch. An interrupted epoch repeats; completed
folds and stages are reused. An interrupted PCA subject is recomputed. Completed
artifact hashes and frozen input/source/environment checks guard against stale reuse.
An interruption can conservatively charge the last unused prepaid minute per device.

## Outputs

- `replay/`: original and ported predictions, fixed-tolerance equivalence reports, and variance checks.
- `results/`: fresh prediction tables, pooled/per-subject/per-category metrics, and variance decomposition.
- `comparison.md` and `comparison.json`: replay and fresh-training comparisons.
- `comparison.png` and `comparison.pdf`: full and residual Figure 9 curves.
- `manifest.json`, `requirements.freeze.txt`, and `budget.json`: provenance and compute accounting.

This implementation and reproduction do not commit, push, or publish repository changes.

## September 14 runner repair

The first fresh fractional-weight experiment trained its encoder successfully, but
the runner looked for `visual_0.25.pt` while the original trainer saved
`visual_0_25.pt`. Automatic retries reached the service restart limit on September 12.
Checkpoint lookup now follows the original trainers' filename sanitization for all
five weights. Result directory names and numerical training behavior are unchanged.
A regression test compares runner names with both trainers.

After the requested reboot pause, the user authorized resuming. All nine equivalence
and recovery tests passed. The repair was applied to the run snapshot after archiving
the previous source, manifest, budget, and completed-stage records under
`revisions/0001_checkpoint_names/` in the run directory. The patch, test log, and
revision metadata are retained there, and the revised manifest records the amendment.
Source, environment, and input verification passed before and after the repair.
The service is enabled again. Completed replay, PCA, baseline, and weight-zero
results are reused, along with the saved weight-0.25 encoder recovery state.
