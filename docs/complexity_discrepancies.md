# Complexity reproduction investigation

For the subsequent RT provenance findings, successful-trial protocol, and corrected
CV investigation, see the [revised comparison](complexity_corrected_protocol.md).
The reproduction results below describe the earlier historical run.

Updated 14 September 2026. The complete generation, fitting, and evaluation flow
executed, but it does not reproduce every published result. This report separates
a repaired migration defect from differences in historical results and fresh
ScanDiff predictions. The original run outputs and posterior files are preserved.

## NSD labels and the Route B reference

The preserved [ranking](../complexity/nsd_m2_ranking.csv) remains the
reference for Route B: 12,447 image-target filenames across 16 target categories.
Its SHA256 is:

```text
2101a2143b00da45625afeaeda08880c70761f825efb0a7c17769624c0a0e6fb
```

The 9 September refit using saved NSD scanpaths reproduced that CSV byte for
byte. Regenerating scanpaths from images produces 12,448 labels: 14 added, 13
missing, and 12,434 shared. Shared scores have Spearman 0.9993988847 and RMSE
0.0101820153. The fresh M2 fit has zero divergences, maximum image-effect R-hat
1.00527, minimum bulk ESS 14,050, and minimum tail ESS 3,922. Numerical and
sampling criteria pass, but the required label identity fails. The verifier now
lists added and missing labels explicitly without relaxing that criterion.

All 524,810 raw NSD records have the same image/observer identities and order as
the historical records. Fixation lengths change in 214 records and initial
success flags change in 219. Filtering leaves 82,211 positive-count trials,
compared with 82,221 historically: 76 are added and 86 removed. Among the 82,135
shared trials, 20 change count.

Each of the 27 changed labels is explained by a single decisive trial under the
existing success filters. Seventeen decisive trials change initial success;
twelve change whether a fixation intersects the corrected target box, with two
trials in both groups. None of these label changes requires a different input
partition, a lost image, or a changed filtering implementation.

For example, `test-0033_nsd-09069_microwave.png`, observer 1, has seven saccades
in both versions. Its closest fixation moves from 0.00052 pixels outside the
corrected box to 0.06198 pixels inside. This creates the image's only qualifying
trial and hence a new label. These are small coordinate differences with a
discrete effect on inclusion.

The original converter assigns each observer's trial index after filtering on
initial success. The changed success flags consequently shift trial indices in
69,752 shared included records, by at most eight. This changes a model covariate
even when an individual scanpath's count is unchanged. That historical behavior
is preserved.

Low trial support explains some larger individual score movements. For
`train-3951_nsd-31953_sink.png`, included trials increase from one to two and the
score changes from 0.54973 to 0.95252, the largest shared-label difference,
0.40279. Overall high correlation does not imply every individual label is stable.

| Variance decomposition | Saved labels | Regenerated labels |
| --- | ---: | ---: |
| Total variance | 0.0722349801 | 0.0722295509 |
| Within-image variance | 0.0340688631 | 0.0341040071 |
| Between-image variance | 0.0381661169 | 0.0381255438 |
| Within-image percentage | 47.16394% | 47.21614% |
| Scenes with at least two targets | 1,205 | 1,206 |
| Averaged scene-target pairs in those scenes | 2,555 | 2,556 |

Controlled original-versus-adapter generation matches exactly in the new
environment, including four-worker partitioning, corrupt-input skips, and random
state restoration. The precise historical software and arithmetic environment
was not recovered. Numerical differences upstream of filtering are measured;
their exact historical hardware or library cause is not established. Saved
scanpaths provide the verified route to the historical labels. Fresh generation
must retain its failed identity result and must not replace the Route B reference.

## COCO generation and count tables

All 31,010 regenerated COCO records retain their historical identities, order,
fixation lengths, and success flags. Coordinates differ by at most 0.12793 pixels
and total RT by at most 0.10363 milliseconds. This preserves count inputs exactly
while slightly changing RT inputs. Only three complete records are exactly equal.

The following comparisons count agreement at the paper's printed precision.
Both saved and freshly generated COCO branches give these counts. Replacing the
human joint RT LOO rows with the corrected scores leaves these totals unchanged.
The RT scoring correction does not affect the count tables or CV results.

| Paper table | Matching cells | Differing cells |
| --- | ---: | ---: |
| Table 1, human RT | 0 | 63 |
| Table 2, human count | 26 | 2 |
| Table 3, generated count | 28 | 0 |
| Table A1, human RT | 0 | 54 |

The two human-count differences are near a rounding boundary, but do not match
the printed values: M3 pair-probability difference is 0.0853489168 versus 0.0854;
M4 is 0.3164465312 versus 0.3165. These differ by about 0.000051 and 0.000053.
Agreement of summary values does not override failed sampling diagnostics below.

## Repaired joint RT LOO migration omission

The original RT script's CLI explicitly called
`compute_joint_marginal_loglik(..., mcN=max(20, args.cv_mcN), seed=args.seed)`
before comparing LOO scores. The migrated shared runner preserved the scientific
functions but omitted this call from the replaced CLI. Its joint LOO rows
therefore conditioned on observed count instead of integrating over predicted
count, making them unsuitable for that comparison with RT-only models.

The [runner](../complexity/models/runner.py) now restores this preparation for all
five joint variants, using the original function without changing its equations.
Failure propagates instead of silently falling back to a different likelihood.
A regression test checks the marginal likelihood and its selection by LOO.
Fitted posteriors, ranking exports, and CV calculations are unaffected. The
original CV implementation already performs marginal prediction.

The [rescoring command](../complexity/evaluation/rescore_rt_loo.py) reads completed
posteriors, verifies input and model-source hashes, and writes to a separate new
directory. It retains seed 42 and the original minimum of 20 count samples.
The three distinct RT suites are rescored separately: historical human,
historical predicted, and regenerated predicted. The regenerated human branch
shares the historical human fits because its input is identical. The initial
migrated run's joint LOO rows are superseded by these separate marginal scores,
without rewriting archived outputs.

All 27 LOO evaluations across the three suites are complete, including 15 joint
evaluations. The 12 RT-only evaluations reproduce their earlier scores within
1e-10. M2-RT has the highest corrected LOO ELPD in all three suites; interpretation
of the alternative models remains subject to the diagnostics below.

Remaining independent models were evaluated in parallel under
`ijcv-rescore-parallel.service`, which completed successfully. Completed per-model
scores from the initial sequential jobs were retained after verifying their
posterior hashes; the continuation records its source hash and exact task list.
Each model used the same seed, inputs, posterior, and marginal-likelihood function
as the sequential command. This parallelized evaluation only and did not refit models.

For example, human M2J LOO ELPD changes from +2,908.20 with the unintended
conditional score to -12,896.72 with the restored marginal score. The paper gives
-19,939. The correction is necessary but does not recover that historical value.
Generated M2J changes from +5,399.19 to -9,577.03 on saved scanpaths.

## RT result provenance, subsequently identified

The LOO omission cannot explain RT-only scores or held-out CV differences. The
human input columns `RT`, `N`, `subject`, `image`, and `trial` exactly match the
recovered `merged_gt.csv`. At the time of this reproduction, AST comparisons
confirmed all 36 RT and 32 count top-level scientific functions/classes matched
the original scripts, excluding the replaced CLI. The subsequent revised run
changes CV draw pairing and adds fold checkpoints, as documented in the revised protocol.

Recovered `projects/complexity/notes.md` records human M2-RT CV Spearman
0.9706911591, Kendall 0.8534799516, and top-20 overlap 0.63. The rerun gives
0.9707132314, 0.8535569261, and 0.63. The paper reports 0.9377, 0.7853, and 0.625.
The later provenance audit recovered the original RT summary tables. Their
image-count fingerprint, historical commands and targeted M1/M2 fits strongly
identify unfiltered synthetic data as the source of Tables 1 and A1, despite
their human-data captions. This supersedes the earlier unspecified-version
hypothesis. The original table-producing posteriors and complete CV provenance
remain missing, so a complete historical replay has not been established.
The supplied manuscript does specify target acceptance 0.95 and four chains
with 2,000 tuning and 2,000 retained draws each, matching the rerun. Uncertainty
about the historical run should not be taken to mean those stated settings are
unknown.

Figure 3a's top-20 overlap changes from 16/20 (0.80) to 15/20 (0.75). The human
RT ranking otherwise has Spearman 0.9999847591 against the saved ranking, RMSE
0.0011943168, and maximum score difference 0.0067894690. The cutoff change is
fully identified:

| Image | Historical RT rank | Rerun RT rank | Historical score | Rerun score | In count top 20 |
| --- | ---: | ---: | ---: | ---: | --- |
| `000000228335.jpg` | 20 | 22 | 0.785726 | 0.781911 | Yes |
| `000000409678.jpg` | 21 | 20 | 0.783773 | 0.786009 | No |

The historical boundary gap is only 0.001953. Small score differences explain the
discrete overlap change; the exact source of the historical human RT fit's
numerical differences is not established. This figure effect is separate from
the larger RT table provenance discrepancy.

## Sampling limitations

The selected M2 count and RT fits pass the recorded image-effect diagnostics,
including the regenerated NSD M2 fit. Several alternative models do not:

- Human M4 count: 2,095 divergences, maximum image-effect R-hat 2.468, minimum
  bulk ESS 5.
- Generated M4 count: maximum image-effect R-hat 3.966, minimum bulk ESS 4,
  despite zero divergences.
- Several joint RT models have image-effect R-hat around 1.53 to 1.73 and
  minimum bulk ESS of 6 or 7. Some human joint fits also have divergences.

The full list is in the audit artifact. Original LOO calculations also emit
Pareto-k warnings. Successful execution and matching a printed number do not
validate these alternative fits or establish the paper's convergence claim.
No priors, model definitions, identity grouping, or sampling settings were tuned
to force agreement during this investigation.

## Repository state and evidence

At the start of this investigation, the committed complexity scientific code
matched the executed source snapshot. Only the component README differed.
The current correction and diagnostic reporting changes are confined to
`complexity/` and complexity documentation. Root README changes and ongoing
Route B files are independent user work. The shared reference hash is unchanged.
The historical COCO filename grouping and generated Mean-N baseline are retained.

Seven scientific regression tests pass. Source-manifest hashes and the preserved
reference are checked separately. No style enforcement tooling was added. The
[machine-readable audit](complexity_audit.json) includes before/after LOO scores,
the corrected table comparison, label changes, and diagnostic failures.

Machine-specific evidence is retained under:

```text
/home/riccardo/data/ijcv-complexity-full-2026-09-10/investigation_2026-09-14/
```

| Artifact | Evidence |
| --- | --- |
| `discrepancy_audit.json` | Trial changes, table counts, input equality, full-fit diagnostic failures |
| `nsd_verification.json` | Explicit added/missing labels, shared score agreement, failed identity check |
| `nsd_label_change_causes.csv` | All 27 decisive trials, old/new success, coordinates, corrected boxes and margins |
| `nsd_changed_labels.csv` | Added/missing label scores and trial support |
| `nsd_added_trials.csv`, `nsd_removed_trials.csv` | Full changed trial membership |
| `nsd_changed_model_inputs.csv` | Shared trials with changed count or trial covariate |
| `nsd_largest_score_changes.csv` | Largest shared-label score changes and trial counts |
| `figure3a_cutoff.json` | Two cutoff images and the old/new overlap calculation |
| `audit_discrepancies.py` | Exact machine-specific analysis used for the audit |
| `rescore_commands.json` | Exact commands for all three RT suites |
| `parallel_rt_loo.py`, `parallel_scoring_tasks.json` | Independent scoring continuation and model task list |
| `*_rt/provenance.json`, `*_rt/loo_table.csv` | Corrected scoring status, source/posterior hashes and scores |
| `corrected_historical/`, `corrected_paper_comparison/` | Small table-input overlay and comparisons using corrected LOO with original CV/count results |
| `finalize_audit.py`, `final_audit.json` | Consolidation script and complete audit copied into the repository |

The [full run record](complexity_full_reproduction.md) documents the venvs,
generation recipe, supervisor, original immutable snapshot, and initial outputs.
