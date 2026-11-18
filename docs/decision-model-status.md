# Decision model: status, choices and plan

Status as of 2026-11-18. This page explains where the decision-model work
stands, why each choice was made, what the architecture and strategy are as a
result, what has been built, how data is handled, and what comes next. The
full research record and the item-by-item backlog are in
[decision-model.md](decision-model.md); this page is the summary you read
first.

## In one paragraph

We are building our own decision model, in the style of TypeSafe's Jev, to
learn how they work and to have the tools on hand. A decision model reads a
piece of state and typed questions about it, runs a transformer once, and
returns a probability for each allowed answer instead of generating text.
Step 0 of six is finished: the interface, the formulas, a zero-training
readout, the calibration instruments, and a first measured baseline. Step 1,
the data, is two thirds done. No model has been trained yet. A check on
2026-11-18 turned up one fact that changes the design of step 2, described
under [What the latest check changed](#what-the-latest-check-changed).

## What a decision model is, briefly

- **Input**: a state (text or JSON) and questions of three types. A `choice`
  picks one of named options, a `score` places the state on an ordered scale,
  and a `noul` is a yes/no.
- **Output**: a probability per allowed answer. Never prose.
- **Why it is fast**: one forward pass, no token-by-token decoding.
- **Why it cannot leave the schema**: the softmax only ever contains the
  options the request listed.
- **What has to be earned**: honesty. Callers write code like
  `if confidence > 0.95: act`, which is only correct if 0.95 means right 95%
  of the time. That property is called calibration, and it comes from training
  and measurement, not from the mechanism.

## The choices, and why

| Choice | Why |
|---|---|
| Build it at all, when others already have | The repo exists to learn and to keep tools on hand. Prior work is a shortcut and a reference point, not a reason to stop. |
| Backbone: Qwen3.5-9B chat for development | Every competitive open decision model sits on the Qwen family, and chat checkpoints beat base ones by about two points in published ablations. 9B is the largest size one rented GPU can fully fine-tune. |
| Qwen3.8-27B for one final run, conditional | Only if the 9B result justifies the repo's first multi-GPU job. |
| Full fine-tuning, no LoRA | Adapters have underperformed in practice for the repo's owner, and GPU cost is not the constraint. The strongest open result so far, Baseten's mammoth-1, is also a full fine-tune. |
| Readout: a pointer head | It scores each option's hidden state against a decision token, so it has no limit on option count. The letter-token readout stops at 62 single-token labels. |
| Loss: cross-entropy plus Brier, then a temperature per question type | Both are proper scoring rules, meaning the best achievable loss is at the true frequency. The best-calibrated open models use exactly this and no reinforcement learning. |
| Measure coverage at a 5% error budget beside calibration error | Calibration error checks the scale of the confidences. Coverage checks their ordering, which is what automation depends on and what temperature cannot fix. |
| Public data from one existing collection | A survey found about 750 public datasets already converted to decision format by tasksource, with cleaner provenance than we would produce. |
| Keep our own generator anyway | It is the only data where we set the label frequencies ourselves, so honesty has a known answer, and it can hide the deciding information on purpose. |
| One branch and one pull request per unit of work | So any piece can be found and reverted later. |

## Architecture as planned

1. **Serialize** a request into tokens: the state, then each question with
   its options wrapped in reserved marker tokens, ending in a decision token.
2. **Run the backbone once** over the state.
3. **Read out** with the pointer head: project the hidden state at the
   decision token to a query, project the hidden state at each option's
   closing marker to a key, take dot products, softmax over that question's
   options only.
4. **Derive** the answer in plain code: the top option, the
   probability-weighted level for a score, and a confidence that rescales the
   top probability against chance.
5. **Train** every weight of the backbone plus the head, one epoch, with
   cross-entropy plus Brier. Fit one temperature per question type on held-out
   data afterwards.

Questions in one request must not influence each other. How that is achieved
is the part that changed this week.

## What the latest check changed

A check on 2026-11-18 covered the repo, the datasets we depend on, TypeSafe's
documentation, and new releases. Four findings, in order of importance.

**1. Our backbone is a hybrid, and that breaks one planned design.**
Qwen3.5-9B has 32 layers: 24 are linear attention, the Gated DeltaNet family,
and 8 are full attention, in a three-to-one pattern. Qwen3.8-27B has the same
pattern over 64 layers. Step 2 planned to isolate questions with an attention
mask inside one sequence. A mask only works in the 8 attention layers. The 24
linear layers carry a running state from left to right through every token,
so a later question would still see an earlier one.

The fix is the design Baseten and Bespoke already use: process the state once
as a shared prefix, keep its cache, and run each question as its own short
continuation from that cache. Isolation then holds by construction, and an
extra question costs one short pass. During training the simplest form is one
question per sequence, which is how the public data is already shaped.

This is also a pleasant coincidence for the repo. The main roadmap's next
step is linear attention, with Gated DeltaNet after it. The model we are
about to fine-tune is the architecture that roadmap is building toward.

**2. A full fine-tune of the same model family has been published.**
Baseten's elephant-1 is Qwen3.5-9B, fully fine-tuned for one epoch with
cross-entropy plus a Brier term at equal weight, which is our planned loss.
Three details differ from our plan and are worth adopting or testing:

| | Our plan | Baseten elephant-1 |
|---|---|---|
| Peak learning rate | 1e-5 to 2e-5 | 3.5e-6, 3% warm-up, cosine decay |
| Unanswerable questions | measured with our hidden-information data | a trained "I don't know" option |
| Attention inside a question | causal | bidirectional in the attention layers, reported as +3.8 points in one run |

Their run used 6.2 million examples on 128 GPUs. Ours will be far smaller,
so the learning rate does not transfer directly, but our range should start
lower than planned. They report no calibration numbers.

**3. Jev moved on.** Jev 1.14 was released on 14 October. It accepts images
and adds a fourth question type, Box. Our research described 1.13. TypeSafe
has still published no paper, weights or training details.

**4. The datasets we depend on are unchanged.** Tasksource is at the
revision our adapter was tested against. Three or more new decision models
appear on the Hub every day, and OpenAI and Google now offer hosted ones.
There is still no single standard benchmark.

## What has been built

Seven pull requests are merged, with 452 tests passing.

| Piece | What it does |
|---|---|
| `decide/schema.py` | The request and response shapes, in TypeSafe's field names so their client libraries and public benchmarks work against us later. |
| `decide/derive.py` | The score and confidence formulas, matching TypeSafe's documented example to the digit. |
| `decide/readout.py` | The zero-training baseline: a lettered prompt, one pass, softmax over the letters. |
| `decide/calibration.py` | Brier, log loss, calibration error with its noise floor, reliability tables, coverage at an error budget. |
| `decide/data/` | The record format, the synthetic generator, and the tasksource adapter. |
| `demo/decide/decision_model.ipynb` | The whole mechanism built by hand on a toy model. |

**The one measured result.** Qwen3.5-9B, untouched, on BoolQ's 3,270
yes/no questions through the zero-training readout:

| Measure | Value |
|---|---|
| Accuracy | 0.868 |
| Calibration error | 0.012, against a noise floor of 0.010 |
| Coverage at 5% error | 0.595 |

The model was already honest on this task, which the hypothesis had
predicted it would not be. Others have since reported the same: a letter
readout on an untuned chat model scores within a point of Jev. So the bar
for our fine-tune is a strong baseline, and a trained model that is less
calibrated than this has broken something.

## Data

Everything is stored in one format, a `Record`: the state, the questions, the
label for each question, and where known, the true distribution the label
came from. One training loop and one evaluation script read all of it.

**Generated data, where honesty has a known answer.** The generator builds
three kinds of state: support tickets as free text, log lines with numbers
and tags, and order records as JSON. Each question's labels are governed by
regions. A region is a rule: when this cue is in the state, draw the label
from this distribution.

- A **deterministic** region fixes the label. An honest model is certain.
- A **stochastic** region leaves it split by a die we chose. Tickets saying
  "charged twice" are billing 60% of the time and account 40%, with identical
  text. An honest model says 0.6.
- An **unrecoverable** corpus withholds one question's cues entirely. The
  label still follows the hidden rule, and the recorded truth is the base
  rate. An honest model reports a flat distribution.

Every record stores the die its label was drawn from. Training never reads
it. Evaluation does, and asks whether the model said 0.6 where we wrote 0.6.
No public dataset can support that test.

**Public data, for real language and volume.** One adapter maps
`tasksource/tasksource-jev-typed-decisions` onto our records. That collection
has about a million rows from roughly 750 public datasets, with a license
class per row. The adapter does three things beyond renaming fields:

- It turns annotator vote shares, present on about one row in eight, into a
  recorded truth.
- It drops the sources that our evaluation benchmark tests on, so that
  benchmark measures sources the model has not seen.
- It sets aside 5% of training for fitting temperatures, split by group, so
  that rows sharing a state never land on both sides.

On a 30,000-row sample no row was malformed and no group crossed a split.

**Evaluation-only data, not yet adapted.** Three sets are never trained on:
jev-bench (22 public sources with human labels and vote shares, the most
trustworthy), typed-decisions (the most-used leaderboard, graded against a
panel of models and not people), and OpenDecisions (which has a published Jev
score).

## Roadmap

| Step | Content | State |
|---|---|---|
| 0 | Interface, formulas, baseline readout, calibration instruments, first number | Done |
| 1 | Data: generator, public adapter, evaluation sets, two baselines | Two of four items done |
| 2 | The pointer head, the serializer, question isolation, the loss | Not started; isolation design to be revised |
| 3 | Full fine-tune of the 9B, temperatures, evaluation, order-bias test | Not started |
| 4 | Experiment: does anything beat cross-entropy plus Brier plus temperature | Not started |
| 5 | A Jev-compatible server, and the conditional 27B run | Not started |

**Next, in order:**

1. Adapters for the three evaluation-only sets.
2. Option shuffling when public data is loaded.
3. Two baselines on the evaluation sets: TF-IDF with logistic regression on
   the Mac, and the 9B zero-training readout on a rented GPU. That closes
   step 1.
4. Step 2, built first on a toy model where every property is testable in
   seconds, then attached to the 9B.

**Decisions waiting on the repo's owner:**

- Replace the mask-based isolation in step 2 with the shared-prefix design.
  This is forced by the backbone and is recommended without reservation.
- Lower the planned learning-rate range for step 3 to start near 3e-6.
- Whether to add a trained "I don't know" option, and whether to try
  bidirectional attention inside a question. Both are optional experiments.
- Whether to add the JevBench coherence suite as a fourth evaluation set. It
  needs no gold labels and tests question isolation directly.

## Sources for the 2026-11-18 check

- Model configurations: `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.8-27B` on Hugging Face.
- [Baseten: mammoth-1 and elephant-1](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)
- [TypeSafe model list](https://docs.typesafe.ai/models.md)
- [Releasing the Inner Jev](https://arxiv.org/abs/2610.17205)
- [tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions)
- [jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench), [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), [OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions)
