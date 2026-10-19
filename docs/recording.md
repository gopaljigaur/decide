# Recording decisions and fitting a threshold

`decide.calibration` fits a temperature or a conformal cut from rows of
`(probabilities, correct index)`. Production is where those rows come from.

```python
from decide import Client, Choice, Gate, JsonlRecorder, record_outcome

client = Client(backends, Gate(), recorder=JsonlRecorder("decisions.jsonl"))

r = client.decide(state, {"tier": Choice("How much effort?", TIERS)}, decision_id=task_id)
# ... later, when you know how it actually went
record_outcome("decisions.jsonl", task_id, {"tier": "complex"})
```

One JSON object per line, append-only, safe from several processes at once. A recorder
that raises is ignored: a full disk must not fail a decision. Write your own `Recorder`
for a database or a log pipeline; it needs one method, `record(row)`.

An outcome is the answer that turned out to be right: an option name for a `choice`, a
bool for a `noul`, a level index or name for a `score`. The last outcome recorded for an
id wins, so a correction can be appended. Decisions with no outcome are ignored by
everything below, since an unlabelled row cannot inform a threshold.

## Reading it back

```bash
decide fit decisions.jsonl tier
```

```
312 labelled rows for 'tier'
confidence         0.1063 to 0.9942 as reported
temperature        1.8740  (above 1 means overconfident here)
conformal cut      0.2214  for 90% coverage

threshold  accepted  share  accuracy when accepted
     0.50       241   0.77                  0.9378
     0.60       198   0.63                  0.9596
     0.70       156   0.50                  0.9744
     0.80       104   0.33                  0.9904
     0.90        51   0.16                  1.0000
```

That table is the decision: it says what each threshold would have accepted and how often
it was right. A confidence range well below every threshold is why a table of zeros
happens, which is worth knowing before concluding the model is useless.

In Python the same three pieces are `rows_from`, `accuracy_by_threshold` and
`recorded_confidences`, and `rows_from` feeds `fit_temperature` and
`fit_conformal_quantile` directly:

```python
from decide import rows_from
from decide.calibration import fit_conformal_quantile

rows = rows_from("decisions.jsonl", "tier")
cut = fit_conformal_quantile(rows, alpha=0.05)  # miss at most 5% of the time
```

## Shadow mode

Record without acting on the answer, compare against what happened anyway, and turn the
gate on only when the table says it is safe:

1. Run the decision on every request, record it, ignore the answer.
2. Attach the outcome your existing system produced, by id.
3. Run `decide fit` and read the accepted share against the accuracy.
4. Set `Gate(min_confidence=...)` to a threshold off that table, or fit a conformal cut
   and use `ConformalGate`.

Step 3 is the one that is usually missing, and it is why thresholds get guessed.

Back to the [README](../README.md).
