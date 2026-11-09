# Zero-shot decisions from Qwen3.5-9B on BoolQ: the step 0 baseline

- **Date:** 2026-11-09
- **Config:** `config.yaml`
- **Hardware:** 1× RTX 5090 (32 GB), RunPod
- **Versions:** PyTorch 2.13.0+cu130, transformers 5.16.1, CUDA 13.0, driver 595.91
- **Commit:** `c447571` (branch `decide/step0-zero-shot`), plus the answer-position logits fix that followed it

## Hypothesis

*Written before the run.* The baseline readout (`litterbox.decide.readout`) on
the untouched Qwen3.5-9B chat checkpoint, framed in its chat template with
thinking off, answers BoolQ's 3,270 validation items as `noul` questions. We
expect accuracy between 0.80 and 0.88, in the range a 9B chat model reaches on
BoolQ zero-shot, and we expect the probabilities to be *dishonest*: ECE well
above its noise floor, with the model overconfident at the top, because
preference-tuned chat models are overconfident and nothing here corrects that.
Coverage at a 5% error budget should be modest, under 0.5. Falsified if
accuracy is near 0.62, the majority-class rate, which would mean the lettered
prompt is not being read as a question, or if ECE lands within three standard
deviations of the floor, which would mean the chat model is already honest and
the calibration part of this project has less to do than the research says.

## Setup

One question per item: state = passage, instructions = "Based on the passage,
is the answer to this question yes? <question>", answered as a `noul`, options
rendered as `A = no`, `B = yes`. Prompt wrapped as a user turn, assistant turn
opened, `Answer:` written as the start of the reply so the next token is the
spaced letter. bf16, batches of 16, right padding. Nothing varies; this is one
number, not an ablation. The dataset is BoolQ rather than Banking77 because
Banking77 has 77 intents and this readout has 62 single-token labels; the
pointer head in step 2 lifts that, and Banking77 is the step 3 comparison.

## Result

`results.json` holds the summary and the reliability table; `predictions.jsonl`
every item's probabilities and label. 3,270 items in 135 s on the 5090,
651k prompt tokens.

| | |
|---|---|
| accuracy | 0.868 (majority class 0.622) |
| Brier / log loss | 0.196 / 0.326 |
| ECE, equal-width bins | **0.012**, noise floor 0.0097 ± 0.0028 |
| ECE, equal-mass bins | 0.019 |
| coverage at 5% error | 0.595 |
| AURC | 0.047 |

Reliability by bin (stated confidence → fraction right, count): 0.56 → 0.57
(150), 0.65 → 0.62 (183), 0.75 → 0.74 (347), 0.86 → 0.86 (567), 0.96 → 0.95
(1,996). Every bin sits on the diagonal within two points. One oddity: 27 items
came out at exactly 0.5/0.5 in bf16, where the tie-break picks "no" and 82% of
them were "yes"; a tie rule or fp32 at the readout would remove the bin.

Two false starts worth recording. The first full run thrashed the 32 GB card
with repeated allocation failures: the readout asked the model for logits at
every position over a 150k-token vocabulary, 5 GB per batch of 16, and the
fix was to gather the hidden states at the answer positions and apply the
output head only there. Before that, `examples/tokenizers.py` shadowed the
`tokenizers` package and broke `transformers` imports when running any script
from `examples/`; it is now `compare_tokenizers.py`.

## Conclusion

*Written after.* The accuracy half of the hypothesis held, 0.868 against a
predicted 0.80 to 0.88. The calibration half was **falsified**: ECE is within
one standard deviation of its noise floor, so on this task, through this
readout, the untouched chat model is already honest, and coverage at 5% error
is 0.60 rather than the predicted under-0.5. The overconfidence the research
attributes to preference-tuned models did not appear here.

Three reasons not to generalise yet, each of which is a later step's job.
BoolQ is a two-option question, and the independent Jev audit found the sign
of miscalibration differs by primitive, with yes/no the best behaved. BoolQ is
also almost certainly in Qwen's pretraining data, so this is in-distribution
calibration; the unrecoverable-label variant in step 1 is the out-of-
distribution test. And the letter readout caps at 62 options, so Banking77's
77-way question, where single tokens strain, is untested until the pointer
head exists. What this run does establish is the baseline the rest of the
roadmap has to beat: 0.868 / 0.012 / 0.595 on BoolQ, and a step 3 model that
does worse than this on calibration has broken something the base model had.
