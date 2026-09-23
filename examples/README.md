# Examples

Runnable scripts showing how to *use* the library.

The distinction from [`demo/`](../demo/): those notebooks build a technique from
scratch to explain how it works. These scripts assume it works and show you the
API. Reach for `demo/` to learn byte-pair encoding; reach for here to find out
how to swap one tokenizer for another.

Most run standalone with no arguments and no network. The quickest tour of the whole stack:

```bash
python examples/fresh_model.py
```

Optional backends degrade to a printed note rather than an error, so nothing
here requires the full dependency set.

## Contents

| Script | Shows |
|---|---|
| [fresh_model.py](fresh_model.py) | The whole stack on an untrained model, in seconds, with nothing downloaded: build from YAML, a forward pass that lands at `ln(vocab)`, generation (nonsense, on purpose), the KV cache matching cache-free generation, and the mixer swapped across three configs with the cache profile changing and nothing else |
| [tokenizers.py](tokenizers.py) | Building tokenizers from config, why special tokens are data rather than behaviour, comparing compression and segmentation, and registering an algorithm of your own |
| [data_pipeline.py](data_pipeline.py) | Text to training batches: packing into shards, what the metadata sidecar is for, deterministic batching, and what reading actually costs. `--big` for a 200MB run |
| [train_tinystories.py](train_tinystories.py) | The whole path end to end: pack TinyStories from a data config, assemble a small dense model, train it with the hand-written loop, and print a story at every evaluation. Needs the network once, to fetch the corpus. Resumes from its last checkpoint; picks CUDA, then Apple's GPU, then CPU |
| [generate.py](generate.py) | Write text from a trained checkpoint: `python examples/generate.py runs/tinystories-dense --prompt "Once upon a time"`. Sampled or greedy, through the KV cache, with `--check` to confirm it matches cache-free generation token for token |

The performance figures these print are hardware-specific. Re-run them on the
machine you actually train on — the serial-versus-parallel tradeoff in packing
flips with core count and multiprocessing start method.
