# Cost before this session's optimizations

Captured from an earlier run under the ORIGINAL pipeline: all steps on
gpt-6-astra, TF-IDF chunk ranking, no trust tiering, no SEARCH_MODEL/
SYNTH_MODEL split. Full traces for these runs are archived in
traces_before_optimization/. Compare against the equivalent questions
from the final --fresh-memory run once it's done.

| question | cost (old pipeline) |
|---|---|
| q01 | $0.6559 |
| q02 | $0.4165 |
| q03 | $1.1935 |
| q04 | $0.0210 |

(smoke_test: $0.0617 -- not a real question, exclude from comparison)
