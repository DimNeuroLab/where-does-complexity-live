# Original Route B reproduction

Checkpoint replay tests port equivalence. Fresh training tests reproduction under recorded settings.

| Visual weight | Run | Pearson | Spearman | Residual Pearson | MAE |
| --- | --- | --- | --- | --- | --- |
| 0 | Reference | 0.499404 | 0.434136 | 0.045810 | 0.193179 |
| 0 | Port replay | 0.499404 | 0.434136 | 0.045810 | 0.193179 |
| 0 | Fresh training | 0.500561 | 0.436335 | 0.049663 | 0.193056 |
| 0.25 | Reference | 0.593909 | 0.549363 | 0.360750 | 0.178447 |
| 0.25 | Port replay | 0.593909 | 0.549363 | 0.360750 | 0.178447 |
| 0.25 | Fresh training | 0.614741 | 0.571343 | 0.403611 | 0.174596 |
| 0.5 | Reference | 0.610637 | 0.569871 | 0.395328 | 0.175412 |
| 0.5 | Port replay | 0.610637 | 0.569871 | 0.395328 | 0.175412 |
| 0.5 | Fresh training | 0.623225 | 0.580794 | 0.419780 | 0.172926 |
| 0.75 | Reference | 0.608604 | 0.562320 | 0.391693 | 0.175624 |
| 0.75 | Port replay | 0.608604 | 0.562320 | 0.391693 | 0.175624 |
| 0.75 | Fresh training | 0.608443 | 0.565504 | 0.390752 | 0.175764 |
| 1 | Reference | 0.612456 | 0.567506 | 0.399459 | 0.175223 |
| 1 | Port replay | 0.612456 | 0.567506 | 0.399459 | 0.175223 |
| 1 | Fresh training | 0.607412 | 0.562107 | 0.391611 | 0.176004 |
