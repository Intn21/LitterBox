# Decision models, and a roadmap for building one here

Research note, 2026-10-05. Part 1 is what Jev is and how it works, separating
what TypeSafe has said from what the open reproductions have shown. Part 2 is a
draft roadmap for our own, in the style of [ROADMAP.md](../ROADMAP.md): steps
that end at a condition.

---

## Part 1 — What Jev is

### The product

TypeSafe AI (San Francisco, founded 2024 by Diogo Almeida, Erik Gafni and
Sasha Sheng; Almeida worked on RLHF, InstructGPT and ChatGPT at OpenAI)
released **Jev** on 15 September 2026 in waitlisted early access, alongside a
$40M seed led by DCVC. They call it the first **System One model**, after
Kahneman: fast, intuitive judgement rather than deliberate reasoning. The name
is from Jevons' paradox, the bet being that cheap decisions get used far more.

Jev does not generate text. A request is a **state** (text or JSON, up to about
32k tokens) and a map of named, typed **questions**. The answer is a typed value
per question with a probability distribution, in one parallel pass, in
70 to 500 ms. Input costs $0.042 per million tokens; output is free.

The three primitives, from the [API reference](https://docs.typesafe.ai/api):

| Type | Question | Criteria | Answer |
|---|---|---|---|
| `noul` | Is this true? | optional `{true: ..., false: ...}` | one number in [0, 1], the yes probability |
| `choice` | Which of these? | map of up to 255 option names to descriptions | `choice`, `probabilities` over options (sum to 1), `confidence` |
| `score` | Where on this ordered scale? | ordered list of 2 to 10 level descriptions | `score`, `probabilities` over levels, `legend`, `confidence` |

Two derived quantities are documented exactly, which matters because they are
not model outputs. We can implement them today and test them against the docs.

- **Score** is the probability-weighted level index:
  `0·0.00 + 1·0.57 + 2·0.43 = 1.43`. It can be fractional.
- **Confidence** is a shape statistic of the distribution, not the top
  probability. For a choice over `n` options,
  `confidence = (p_max − 1/n) / (1 − 1/n)`, so uniform is 0 and a point mass is 1.
  For a score it penalises distance from the modal level `m`:
  `confidence = max(0, 1 − Σ pᵢ·|i − m| / MAD_uniform)`, so mass on an adjacent
  level costs less than mass on the far end.

Three documented behaviours shape the architecture (see
[primitives](https://docs.typesafe.ai/primitives.md) and
[jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)):

1. **Questions are independent.** Every question sees the same state; no
   answer is context for another. Adding a question never changes an existing
   answer. Adding questions adds little latency, but tokens are billed per
   question.
2. **The schema is closed by construction.** The model cannot return a value
   outside the supplied options. TypeSafe's "0% hallucination" means exactly
   this and nothing more: being confidently wrong *within* the schema is
   common, and the docs say so.
3. **It reads literally and cannot count.** Counting, arithmetic, date
   ordering, double negatives and multi-hop questions degrade it. Option order
   affects the answer, and 1.13 leans toward the first option. Instructions
   embedded in the state can steer it. Irrelevant state lowers accuracy.

### What TypeSafe has disclosed about the internals

Very little, and deliberately. From the
[launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev):
"a new model architecture, parallel sampler for maximum efficiency, and a
training method we call Reinforcement Learning for Calibrated Decisions
(RLCD)." Elsewhere: transformer-based, trained entirely on synthetic data, built
on open-weight models (Financial Times), "neither small nor an LLM." No paper,
no weights, no parameter count. Asked on Hacker News whether it is an encoder
with heads or a text-diffusion model, they answered "staying quiet for now, a
paper may follow later."

RLCD is described only by its goal: probabilities are "optimized against
outcomes rather than against human rater preference," so that answers given 0.9
are right about 90% of the time. Their
[ML primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md)
frames the motivation as RLHF's **mode dropping**: preference optimisation
narrows the distribution toward what raters like, which makes a model
overconfident and inconsistent, and "if a model can do a task 95% of the time
but doesn't say when it's in the 5%, it can't automate that task."

### What the open reproductions have shown

Within three weeks, four public reimplementations converged on one mechanism.
The ideas are old (zero-shot NLI, cross-encoders, reward models, BERT
classification heads); what is new is defining the classes at request time and
training explicitly for calibration.

**The mechanism: prefill, then score; never decode.** Serialize state and
questions into one prompt. Run a single forward pass. Read the logits for the
allowed answers at one position. Softmax over *only those* logits. That is the
whole speed story: a decode loop of hundreds of steps becomes one prefill, which
is the operation GPUs are best at. It is also the whole schema story: a softmax
over the supplied options cannot produce anything else.

Two ways to read the logits:

| Readout | Who | How |
|---|---|---|
| **Single-token labels** | [Jebadiah](https://github.com/getainode/jebadiah), several OpenJev variants, the Together AI tutorial | Options are rendered as `A`, `B`, `C`...; the standard LM head's logits for those tokens at the answer position are the option scores. No new parameters beyond LoRA. Capped by how many single-token labels exist (Jebadiah: 68). |
| **Option markers plus a head** | [Clef](https://blog.cloudflare.com/clef-decision-models/) ("joint schema head"), [Laya](https://huggingface.co/convaiinnovations/laya) (`[MASK]` per option), [open-jev](https://github.com/kyegomez/open-jev) (query slots cross-attending into the encoded state) | Each option gets a position in the input; a small head scores every option of every question jointly from the backbone's final hidden states. Scales to many options and to many questions in one pass. |

Clef's head is the most detailed public description: a small transformer that
"reads the backbone's final hidden states, routes evidence from the state to
each question, and scores all options of all questions jointly," with a
two-stage attention where each option extracts relevant context and fields can
cross-attend to each other and back to the payload before scoring. The backbone
(Qwen3.8-27B, or Qwen3.5-9B for Clef-flash) is frozen; the head trains jointly
with rank-256 LoRA adapters.

**The training objective: proper scoring rules, then temperature.** Everyone
uses a strictly proper scoring rule, which is the mathematical reason honest
probabilities maximise reward:

- Clef: label-smoothed cross-entropy plus a **Brier** term; an RL stage they
  also call RLCD that gives partial credit to adjacent score levels, rewards
  fully correct multi-question records, and adds a reference penalty against
  drift.
- Laya: REINFORCE with a group-mean baseline (GRPO-style) against log and
  spherical scores, plus the **ranked probability score** for ordinal
  questions.
- Jebadiah: plain cross-entropy over the candidate logits, with an ordinal
  kernel for score questions (adjacent levels weighted 0.2). No RL at all.
- Verdict: cross-entropy plus Brier, with an explicit `__insufficient_evidence__`
  option for abstention.

Every one of them then fits a **temperature** per question type on a held-out
slice. The reported effect is large: Laya's ECE goes from 0.466 to 0.081 with
temperature alone; Verdict's optimal temperature was 1.43. Jebadiah's recipe,
which skips RL entirely, reaches ECE 0.014 on the Decision Index, better than
Jev's 0.074. The lesson is that **calibration comes mostly from the scoring
rule and the temperature fit**, and that RL is the uncertain part of the recipe.

**The data: typed-decision records, with the structure permuted.** Clef trains
on "internal synthetic datasets permutating field orders, prompts, and schema
structures"; open-jev trains consistency across paraphrased and key-shuffled
states. Jebadiah's data is public and license-checked: `typed-decisions`,
BoolQ, MNLI, DBpedia14, HelpSteer2, SummEval, with a manifest of every source.
Their synthetic pools from teacher models "showed minimal gains," while
starting from the *chat* checkpoint rather than the base gave 2 points for free.
Verdict found the opposite of robustness: random option permutations flip 4.5%
of answers, and abstention collapses from 75% recall to 18% under hard sibling
options.

### How good is it, really

Independent numbers, as of early October. Jev is the strongest zero-shot
generalist among decision models and is level with mid-price LLMs, but it is
behind the frontier on accuracy, and its calibration is good in distribution
and poor out of it.

| Study | Result |
|---|---|
| Ibrahim & Zaki, 15 social-science tasks, 7,977 human labels | Jev a median 11.6 macro-F1 behind the best LLM per task; behind on 14 of 15 |
| Banking77, 8 independent runs | Jev 75.3 to 84.0%, median 80.9%; Clef 94.2% |
| Bespoke Labs, 13 public subsets | Jev median ECE 0.071, better than Gemma 4 variants (0.114 to 0.180) |
| [jev-ood-calibration](https://github.com/scienthoon/jev-ood-calibration), 900 rule-generated tickets | choice 89% / ECE 0.082, boolean 92% / ECE 0.079, score 45% / ECE 0.325; the unknowable score task got 0.74 mean probability on the chosen level |
| Decision Index board | Jev 1.13 57.91; pplx-decider 27B 56.40 (ECE 0.018); Jebadiah 27B 54.67 (ECE 0.014) |
| Speed, 3,080 messages | Jev 245 ms median vs 249 ms for Qwen3-Coder-Next-80B; Clef-flash 39 ms |

The sign of miscalibration differs by primitive on the same inputs: choice and
score are overconfident, noul underconfident. That is evidence of three readouts
sharing one backbone rather than one uniform mechanism.

The honest summary of the 193.6×/444.6× marketing claims: real on decomposed
workflows with many short questions, where output tokens dominate LLM cost;
one independent attempt put the realistic figure near 25×; and TypeSafe's own
benchmark measures agreement with other models, not ground truth.

### Where the accounts disagree

Added 2026-10-05 after a second pass over English, Chinese and Japanese
sources. The mechanism above is the consensus, but it is not unanimous, and
the disagreements cluster in four places.

**1. Whether the mechanism was ever the point.** The loudest dissent is that
there is nothing to reverse-engineer. The r/LocalLLaMA and Hacker News line is
that Jev is a zero-shot classifier, a category BERT, GLiNER and NLI
cross-encoders have covered for years; a Japanese reviewer who read seven
clones traces the readout to PET (2020) and the recalibration to "Calibrate
Before Use" (2021), and concludes the 2026 surge "fitted existing techniques
into that API shape." A CSDN essay calls it "带智商的高速 if-else", intelligent
high-speed if-else. Han Xiao's reaction was surprise that a discriminative
model could excite the public at all. TypeSafe's founder concedes the point on
architecture and relocates the claim: on the Latent Space podcast Almeida calls
the company "a data lab rather than a model lab," says 100% of the data is
synthetic, and describes the work as finding the model's "jaggednesses" and
addressing them "surgically." He declines every architectural question. The
quote "exactly right!" in reply to "basically a zero-shot classifier" is
widely repeated and could not be located in the launch thread.

**2. Whether the open clones actually reproduced it.** The two-hour
reproductions that circulated on Zhihu ("网友用两小时就实现了 TypeSafe 用了两年
RLCD 才做到的效果") are stock Qwen2.5-1.5B-Instruct with a parallel
constrained-decoding trick: one batched pass fills every field of a closed
schema and reads per-field probabilities. No training, no calibration. They
reproduce the latency and the schema closure, which is the part that was
never hard. Two measurements say the trained part did not transfer. On
JevBench's calibration axis Jev scores 82.7 and the clones 42.0 to 72.6. And a
Kev-9B that matches Jev on ECE (0.042 vs 0.049) automates half as much traffic
at a 5% error budget (coverage 0.45 to 0.57 vs 0.70). ECE measures the scale of
the confidence; coverage at fixed error measures its *ordering*, and
temperature scaling is monotone, so it cannot fix ordering. That ordering is
the one number where Jev still leads every open model, and it has to have come
from training data, which is the part TypeSafe keeps.

**3. What the readout is.** Three families now exist, and the API evidence
picks between them. Archer Hume's reverse-engineering from about 10,000 calls
is the only systematic probe: latency is linear in state length and sublinear
in question count (1,500 questions in under 600 ms); a 255-option question
returns as fast as a two-option one; a code hidden in a sibling question is
invisible (probability 0.00) and visible when moved to the state (0.90); and
adding an irrelevant option shifts the log-odds between the existing options
by −0.28, which fixed independent logits cannot do. The conclusion is a causal
backbone, likely sparse MoE given 30k tokens in 160 ms, one shared-state
prefill, isolated question branches, and a **listwise** readout where the
decision position attends to the whole option list before scoring. That is the
pointer-head design (a `<decide>` query dotted against each option's last
hidden state) that Kev and AWS's Strands Decider adopt, and it is not the
single-token-label readout Jebadiah uses. Jev's option-order bias, admitted in
its own jaggedness page, is the predicted side effect of a causal listwise
readout. Hume also found Jev's tokenizer matches none of 192 public ones,
closest to Qwen at 348/415 probes, which argues against a straight open-weight
fine-tune.

The diffusion hypothesis has a different status. DiffusionGemma with
constrained logits agrees with Jev on about 90% of answers and Matt Mastracci's
live evals had it "roughly tied in intelligence," which some read as evidence
Jev is diffusion-based. It is not: agreement on answers is what any strong
classifier shows, and the latency curves above are the prefill-and-readout
shape, not iterated denoising.

**4. What RLCD contains.** Nobody outside TypeSafe knows, and three accounts
circulate. Clef and Laya publish RL stages (partial credit for adjacent
levels, group-baseline REINFORCE against log and spherical scores). Jebadiah
publishes none and calibrates best. A widely shared Japanese "deep research
report" gives RLCD specifics (Gaussian logit noise decaying from 1.0 to 0.3,
eight sampled candidates, a log-score floor of −9.21) that match no TypeSafe
statement and could not be traced to any clone's documentation; treat them as
invented. The closest published prior art is "Rewarding Doubt"
(Bani-Harouni et al., 2025), PPO against the log scoring rule for verbalised
confidence. One substantive critique of the *scope* of RLCD comes from the same
Japanese report and from Anthony Maio: calibration is measured per call on
independent items, and nothing in TypeSafe's material says it survives
composition, where one miscalibrated decision becomes the state of the next.

**The field three weeks on.** OpenAI's Decisions API on GPT-6 Luna returns a
self-reported confidence that independent tests found uncalibrated (99% meant
68% on one suite). Alibaba's Bailian `decision-model-preview` speaks the
System One protocol, discloses nothing about its model, and the first V2EX
thread about it shows a 50/50 answer at confidence 0.01 on "delete all rows
from the users table" that flipped to "safe" at 0.69 when "no backup" was
added. AWS's Strands Decider 2B is the cleanest open statement of the
pointer-head design, 19 iterations in, and reports that an earlier "slot head"
was significantly worse. Perplexity's pplx-decider 27B and Bespoke's Nimble
round out the board.

What this changes in Part 2: step 2 should build the **listwise pointer
readout** rather than single-token labels, since that is what the evidence
says Jev does and what the order-bias test exercises; and the evaluation in
step 4 should report coverage at a fixed error budget alongside ECE, because
that is the number the clones have not matched.

### Why it matters for this repo

A decision model is a transformer with the LM head replaced by a schema-closed
readout and the cross-entropy objective replaced by a calibration objective.
Everything else, the backbone, the training loop, the data packing, is what we
already have. The parts we do not have, a bidirectional or block-masked
attention pattern and a non-LM head, are exactly the kind of seam this repo
was built to test.

---

## Part 2 — Draft roadmap: a decision model in litterbox

Revised 2026-10-05 after the dissent pass. Two things changed: the readout is
listwise (a decision position that sees the whole option list), because that
is what the API evidence says Jev does; and the headline calibration metric is
coverage at a fixed error budget, because that is the number no open clone has
matched and temperature scaling cannot fix.

**The goal, stated as something testable:**

> A `/v1/systemone`-compatible server backed by a model we trained, which on a
> held-out typed-decision set beats the TF-IDF + logistic-regression baseline
> on accuracy, reports probabilities with ECE under 0.05 after temperature
> fitting, and automates more traffic at a 5% error budget than the same
> model with its confidences shuffled — with schema closure and question
> isolation proven by tests rather than claimed.

Numbers to beat, from the public record: on `typed-decisions` the TF-IDF
baseline scores 66.1% accuracy at ECE 0.021; Jebadiah 4B scores 72.5%, Laya
base zero-shot 36.2%, Jev 72.7%. On coverage at ≤5% error, out of domain, Jev
reaches 0.70 and Kev-9B 0.45 to 0.57.

### Two routes, and why we take both

- **Route A, a pretrained backbone.** LoRA on a small open chat checkpoint,
  prefill, read out, cross-entropy plus Brier, temperature. Jebadiah's recipe
  with Kev's readout. A *useful* model in a few GPU hours.
- **Route B, our own backbone.** The same readout and loss on the TinyStories
  model, trained from scratch on rule-generated data. A *small* model that
  teaches the mechanism, where isolation, closure and order bias are testable
  with `torch.allclose` on the Mac.

Build B first, then swap in A when the question becomes accuracy rather than
correctness: the repo's reference-tier-then-fast-tier rule applied to a model
instead of a kernel. The readout, the mask and the loss are shared code; only
the backbone differs.

### Step 0 — The interface and the formulas

- [ ] `decide/schema.py`: pydantic models for the request (`state`,
      `questions` of three types with the documented limits: 255 options,
      2 to 10 levels) and the response.
- [ ] `decide/derive.py`: `confidence` and `score` from a probability
      vector, matching the documented formulas and worked examples.
- [ ] `decide/readout.py`, baseline form: given any causal LM and a request,
      render the prompt with `A=`, `B=`, `C=` labels, run one forward pass,
      softmax over the label-token logits. Works on our TinyStories
      checkpoint and on a Hugging Face model through the same function. This
      is the zero-training baseline every later step is compared against.
- [ ] A calibration module: Brier, log loss, ECE with equal-width and
      equal-mass bins, a reliability diagram, a **noise floor** (the ECE a
      perfectly calibrated model would show on the same predicted
      distributions, by resampling), and **coverage at a fixed error budget**
      with its area version (AURC). The last two measure ordering, which ECE
      does not.

> **Condition.** Every answer is in the schema, for any input, by
> construction: a property test that fuzzes states and schemas and never finds
> an out-of-schema value. The confidence and score functions reproduce the
> docs' examples (`1.43`, confidence `0.35`). The ECE of synthetic
> perfectly-calibrated predictions lands on the noise floor, and coverage of
> perfectly-ordered predictions equals 1 minus the error budget's share.

### Step 1 — The data

- [ ] A rule-generated typed-decision corpus: templated support tickets, log
      lines and JSON records with known labels for queue (choice), urgency
      (score), and a handful of boolean properties (noul). Rule-generated
      means we set the label frequencies: "charged twice" is billing 60% of
      the time by construction, so the honest answer is known. A deliberately
      *unrecoverable* variant, where the label depends on a policy absent
      from the state, tests whether the model reports a flat distribution.
- [ ] Public sets through the existing data configs: Banking77 (77-way
      choice), BoolQ (noul), SST-5 and HelpSteer2 (score), ChaosNLI for
      multi-annotator soft labels. A manifest with source and license, like
      Jebadiah's.
- [ ] Augmentation at pack time: shuffle option order, shuffle JSON keys,
      paraphrase instructions, so position is not learnable.

> **Condition.** `litterbox-pack` produces shards for the decision task in the
> same format training already consumes; a held-out split exists for every
> source; the TF-IDF + logistic-regression baseline and the step-0 label-token
> baseline are both run on it and recorded in `experiments/`.

### Step 2 — The listwise readout on our backbone

The token layout, following Kev and Strands Decider:

```
<state> … state … <q> instructions <opt> option 1 </opt> <opt> option 2 </opt> … <decide>
```

- [ ] A block mask on `full_attention`, through the same `_allowed(q_pos,
      k_pos)` seam `sliding_window` uses: the state attends causally to
      itself; each question attends causally to the state and to itself;
      nothing attends across questions. Position ids restart after the state
      for every question, so a question's answer does not depend on how many
      questions precede it. Attention stays causal; no bidirectional mode is
      needed.
- [ ] Reserved tokens `<q>`, `<opt>`, `</opt>`, `<decide>` in the tokenizer,
      so option boundaries cannot be forged by text in the state.
- [ ] A `PointerHead`: project the hidden state at `<decide>` to a query and
      the hidden state at each `</opt>` to a key, scaled dot product, masked
      softmax over that question's options. About a million parameters.
      Because `<decide>` comes last it has seen every option, which is what
      makes the readout listwise and what the 255-option cap no longer
      constrains.
- [ ] Training loop support for a non-LM loss: cross-entropy plus Brier,
      with the ordinal kernel for score questions (adjacent levels get partial
      target mass).
- [ ] Temperature fit per primitive on a held-out slice.

> **Condition.** Four tests pass. *Isolation:* a request with k questions
> returns the same probabilities as k single-question requests, to float
> tolerance, and a code planted in a sibling question is invisible while the
> same code in the state is visible. *Closure:* the softmax has exactly as
> many entries as options, by construction. *Order bias:* measured, not
> assumed — permuting options flips the top answer on fewer than 2% of
> held-out items, and the flips concentrate in low-confidence items. *Overfit:*
> one batch trains to zero loss. Then, on the rule-generated held-out set,
> reported probabilities match the frequencies we wrote into the generator,
> and post-temperature ECE is under 0.05. Record the run in `experiments/`.

### Step 3 — Calibration training beyond cross-entropy

This is the step where the field is least settled, so it is an experiment,
not a feature.

- [ ] Implement the alternatives as loss options: Brier only, log plus
      spherical, ranked probability score for score questions, and a
      REINFORCE/GRPO-style stage with a calibration reward and partial credit
      for adjacent levels (the Clef and Laya recipes).
- [ ] Run the matrix on the step-2 model against the unrecoverable-label
      variant from step 1, where the right answer is a flat distribution, and
      report ECE *and* coverage at 5% error for each.

> **Condition.** A results table in `experiments/` answering one question: does
> anything beat cross-entropy plus Brier plus temperature on out-of-distribution
> ECE or on coverage, and by how much? A negative result closes the step.

### Step 4 — A pretrained backbone

- [ ] Load a small open chat checkpoint (Qwen3.5-0.6B to 4B) through
      `eval/external.py`, attach rank-16 LoRA, and train the step-2 pointer
      head with the step-2 loss on the step-1 data. Runs on RunPod; the 4B
      fits one 4090.
- [ ] Evaluate on the public held-out sets and on JevBench and the Nimble
      public subsets, so the numbers are comparable to the published ones.
      Report accuracy, ECE and coverage at 5% error, against the step-0
      label-token baseline on the same backbone.

> **Condition.** Beats the TF-IDF baseline on `typed-decisions`, lands within
> a few points of Jebadiah 4B (72.5%) with ECE under 0.05, and the pointer
> head beats single-token labels on Banking77, where 77 options is where
> single tokens strain. Coverage at 5% error is reported next to Kev's
> 0.45 to 0.57 and Jev's 0.70.

### Step 5 — Serve it

- [ ] A `/v1/systemone` endpoint speaking the TypeSafe request and response
      format, so the official SDKs and the public benchmark repos
      (`jev-ood-calibration`, `jevbench`) run against it unchanged.
- [ ] Batched prefill with the shared-state block mask, so that adding a
      question costs a few milliseconds, which is the property the product is
      built on.

> **Condition.** `jev-ood-calibration` runs end to end against our server and
> produces its report; latency per added question and per added option is
> measured and recorded, and the per-option curve is flat.

### What stays deferred

Images; a thinking budget before the readout (decode N tokens, then read out
— it restores the decode loop and is a separate experiment on which question
types it helps); abstention as a first-class primitive; calibration under
composition, where one decision becomes the next state; anything over 10
score levels. Each has a trigger; none is on the path to the goal above.

### Relationship to the main roadmap

This does not touch step 6 (linear attention). It uses full attention only,
causal, and the one architectural change it needs, a block mask with
restarted positions on `full_attention`, is the same seam `sliding_window`
already uses.

---

## Sources

TypeSafe: [launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
[API reference](https://docs.typesafe.ai/api),
[confidence](https://docs.typesafe.ai/confidence),
[score](https://docs.typesafe.ai/primitives/score.md),
[ML primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md),
[jev-1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md),
[docs index](https://docs.typesafe.ai/llms.txt).
Coverage: [Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model)),
[TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/),
[Towards Data Science](https://towardsdatascience.com/jev-vs-llms-when-ai-moves-from-generation-to-decision-making/),
[flaviocopes on Jev](https://flaviocopes.com/jev/), [on Clef](https://flaviocopes.com/clef/),
[Victor Dibia](https://victordibia.com/explainers/jev/),
[Laurence Moroney](https://laurencemoroney.com/2026/10/02/decision-models-explained.html),
[eight days of independent tests](https://dev.to/gde/jev-after-eight-days-of-independent-tests-level-with-mid-price-llms-behind-the-frontier-1kln),
[HN: OpenJev](https://news.ycombinator.com/item?id=49752041).
Open models and benchmarks: [Cloudflare Clef](https://blog.cloudflare.com/clef-decision-models/),
[Clef-flash card](https://huggingface.co/Cloudflare/clef-flash),
[Jebadiah](https://github.com/getainode/jebadiah),
[Laya](https://huggingface.co/convaiinnovations/laya),
[open-jev](https://github.com/kyegomez/open-jev),
[Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev),
[jev-ood-calibration](https://github.com/scienthoon/jev-ood-calibration),
[jevbench](https://github.com/dhruvmehra/jevbench).
Dissent and reverse-engineering: [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/),
[Latent Space interview](https://www.latent.space/p/jev), [36kr interview](https://eu.36kr.com/en/p/3994032312630020),
[seven clones, computational form only](https://note.com/zephel01/n/ne9a2c037e513),
[confidence ordering vs temperature](https://saulius.io/blog/jev-rlcd-decision-model-calibrated-probabilities),
[is Jev just a classifier](https://systemonemodels.org/guides/is-jev-just-a-classifier/),
[Arcturus Labs](https://arcturus-labs.com/blog/2026/09/21/will-openai-eat-jevs-lunch/),
[Anthony Maio](https://anthonymaio.substack.com/p/jev-the-language-model-that-wont),
[Kev](https://github.com/sashankh/kev), [Strands Decider](https://strandsagents.com/blog/introducing-strands-decider/),
[Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD),
Zhihu: [两小时复刻](https://www.zhihu.com/question/2084281918074385700), [RLCD 如何理解](https://www.zhihu.com/question/2086022553726990200), [深度拆解](https://zhuanlan.zhihu.com/p/2085057727869593002);
[CSDN 确定性小模型](https://damodev.csdn.net/6aace8bc5c13c42b539d14fe.html), [gm7 过度宣发透视](https://www.gm7.org/archives/159667),
[V2EX on Alibaba's model](https://www.v2ex.com/t/1245606), [Bailian decision model](https://help.aliyun.com/zh/model-studio/decision-model/).
