# Examples

Runnable scripts showing how to *use* the library.

The distinction from [`demo/`](../demo/): those notebooks build a technique from
scratch to explain how it works. These scripts assume it works and show you the
API. Reach for `demo/` to learn byte-pair encoding; reach for here to find out
how to swap one tokenizer for another.

Every script runs standalone with no arguments and no network:

```bash
python examples/tokenizers.py
```

Optional backends degrade to a printed note rather than an error, so nothing
here requires the full dependency set.

## Contents

| Script | Shows |
|---|---|
| [tokenizers.py](tokenizers.py) | Building tokenizers from config, why special tokens are data rather than behaviour, comparing compression and segmentation, and registering an algorithm of your own |
