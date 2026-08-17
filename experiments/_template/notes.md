# <experiment title>

- **Date:**
- **Config:** `config.yaml`
- **Hardware:** <GPU, count, memory>
- **Versions:** PyTorch <x.y.z>, Triton <x.y.z>, CUDA <x.y>
- **Commit:** <sha>

## Hypothesis

*Written before the run.* One paragraph: what you expect to happen, and what
observation would falsify it. If you cannot name a falsifying observation, the
experiment is not yet specified.

## Setup

What varies and what is held constant. For an ablation, state explicitly what
"matched budget" means here — parameters, tokens, wall-clock, or FLOPs — since
the four give different answers.

## Result

Numbers and plots. Point at `results.json` rather than retyping it; write down
what the numbers show, including whatever you did not expect.

## Conclusion

*Written after.* Was the hypothesis supported? What is the next experiment this
implies? If the run failed for a boring reason, say that too — a documented dead
end saves the next person a day.
