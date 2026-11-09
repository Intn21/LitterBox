# Zero-shot decisions from Qwen3.5-9B on BoolQ: the step 0 baseline

- **Date:** 2026-11-09
- **Config:** `config.yaml`
- **Hardware:** 1× RTX 5090 (32 GB), RunPod
- **Versions:** PyTorch 2.13.0+cu130, transformers 5.16.1, CUDA 13.0, driver 595.91
- **Commit:** filled in after the run

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

See `results.json` (summary and reliability table) and `predictions.jsonl`
(every item's probabilities and label).

## Conclusion

*Written after.*
