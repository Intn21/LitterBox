"""Pretraining data pipeline: packing, metadata, and batching.

The properties worth asserting here are the ones whose violation is *silent* —
a wrapped token id, a stale shard, targets that leak the answer. None of those
raise on their own, so each gets a test.
"""

from __future__ import annotations

import numpy as np
import pytest

from litterbox.data import PackedDataset, choose_dtype, pack
from litterbox.data.pack import PackMeta
from litterbox.data.source import build_source, from_files, from_jsonl, from_text
from litterbox.data.tokenizer import Tokenizer, build_tokenizer

GPT2_CFG = {"type": "tiktoken", "encoding": "gpt2"}
BYTE_CFG = {"type": "byte"}


def docs(n: int = 100, repeat: int = 3) -> list[str]:
    return [f"Document {i}. The cat sat on the mat. " * repeat for i in range(n)]


@pytest.fixture
def byte_tok() -> Tokenizer:
    return build_tokenizer(BYTE_CFG)


@pytest.fixture
def packed(tmp_path, byte_tok):
    """A small packed corpus, deliberately sharded so seams get exercised."""
    pack(docs(), byte_tok, tmp_path, tokenizer_cfg=BYTE_CFG, shard_tokens=1000, n_workers=1)
    return tmp_path


# ------------------------------------------------------------------- dtype


@pytest.mark.parametrize(
    ("vocab", "expected"),
    [
        (256, "uint16"),
        (32_000, "uint16"),
        (50_257, "uint16"),  # GPT-2 — the one that fits
        (65_536, "uint16"),  # exactly 2**16 ids, max id 65_535
        (65_537, "uint32"),  # one more, and it does not
        (128_256, "uint32"),  # Llama-3
        (200_019, "uint32"),  # o200k
    ],
)
def test_dtype_follows_vocab_size(vocab, expected):
    assert choose_dtype(vocab).name == expected


def test_dtype_rejects_nonsense():
    with pytest.raises(ValueError):
        choose_dtype(0)
    with pytest.raises(ValueError):
        choose_dtype(2**33)


def test_oversized_ids_are_refused_not_wrapped(tmp_path):
    """A vocab-exceeding id must raise, never wrap modulo 2**16."""

    class Liar(Tokenizer):
        def __init__(self):
            self.name, self.vocab_size = "liar", 300
            self.bos_id = self.eos_id = None
            self._validate()

        def encode(self, text):
            return [70_000] if text == "boom" else list(text.encode("utf-8"))

        def decode(self, ids):
            return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    with pytest.raises(ValueError, match="wrap"):
        pack(["boom"], Liar(), tmp_path, n_workers=1)


# -------------------------------------------------------------------- pack


def test_pack_writes_shards_and_meta(tmp_path, byte_tok):
    meta = pack(docs(50), byte_tok, tmp_path, tokenizer_cfg=BYTE_CFG, shard_tokens=500, n_workers=1)

    assert (tmp_path / "meta.json").exists()
    assert meta.tokenizer_name == "byte"
    assert meta.dtype == "uint16"
    assert meta.tokenizer == BYTE_CFG

    info = meta.splits["train"]
    assert info.n_documents == 50
    assert info.n_tokens == sum(s.n_tokens for s in info.shards)
    assert len(info.shards) > 1, "shard_tokens=500 should have forced several shards"

    for s in info.shards[:-1]:
        assert s.n_tokens == 500, "only the final shard may be short"


def test_meta_round_trips(packed):
    meta = PackMeta.load(packed / "meta.json")
    reloaded = PackMeta.load(packed / "meta.json")
    assert meta.to_json() == reloaded.to_json()


def test_token_count_matches_bytes_on_disk(packed):
    meta = PackMeta.load(packed / "meta.json")
    dtype = np.dtype(meta.dtype)
    for s in meta.splits["train"].shards:
        assert (packed / s.file).stat().st_size == s.n_tokens * dtype.itemsize


def test_two_splits_share_one_meta(tmp_path, byte_tok):
    pack(docs(20), byte_tok, tmp_path, split="train", tokenizer_cfg=BYTE_CFG, n_workers=1)
    meta = pack(docs(5), byte_tok, tmp_path, split="val", tokenizer_cfg=BYTE_CFG, n_workers=1)
    assert set(meta.splits) == {"train", "val"}
    assert meta.splits["val"].n_documents == 5


def test_mixing_tokenizers_in_one_directory_is_refused(tmp_path, byte_tok):
    pack(docs(10), byte_tok, tmp_path, tokenizer_cfg=BYTE_CFG, n_workers=1)
    other = build_tokenizer(GPT2_CFG)
    with pytest.raises(ValueError, match="packed with"):
        pack(docs(10), other, tmp_path, split="val", tokenizer_cfg=GPT2_CFG, n_workers=1)


def test_workers_produce_identical_output(tmp_path):
    """The pool is an optimisation; it must not change a single token."""
    tok = build_tokenizer(GPT2_CFG)
    serial, parallel = tmp_path / "a", tmp_path / "b"
    pack(docs(200), tok, serial, tokenizer_cfg=GPT2_CFG, shard_tokens=5000, n_workers=1)
    pack(docs(200), tok, parallel, tokenizer_cfg=GPT2_CFG, shard_tokens=5000, n_workers=4)

    a = PackedDataset(serial, seq_len=32)
    b = PackedDataset(parallel, seq_len=32)
    assert len(a) == len(b)
    assert np.array_equal(a.read(0, len(a)), b.read(0, len(b)))


# ----------------------------------------------------------------- oracles


def test_packed_stream_decodes_back_to_the_source(tmp_path, byte_tok):
    """Packing must not scramble anything: decode the stream, get the text back."""
    source = docs(10)
    pack(source, byte_tok, tmp_path, tokenizer_cfg=BYTE_CFG, shard_tokens=200, n_workers=1)
    ds = PackedDataset(tmp_path, seq_len=16)

    stream = ds.read(0, len(ds))
    text = byte_tok.decode([i for i in stream.tolist() if i < 256])
    for doc in source:
        assert doc in text


def test_every_document_appears_exactly_once(tmp_path, byte_tok):
    source = docs(30)
    pack(source, byte_tok, tmp_path, tokenizer_cfg=BYTE_CFG, shard_tokens=300, n_workers=1)
    ds = PackedDataset(tmp_path, seq_len=8)

    offsets = ds.document_offsets()
    assert offsets is not None
    assert len(offsets) == len(source)

    ends = [*offsets[1:].tolist(), len(ds)]
    for doc, start, end in zip(source, offsets.tolist(), ends, strict=True):
        ids = ds.read(start, end - start)
        assert byte_tok.decode([i for i in ids.tolist() if i < 256]) == doc


def test_no_tokens_are_lost(tmp_path, byte_tok):
    source = docs(25)
    meta = pack(source, byte_tok, tmp_path, tokenizer_cfg=BYTE_CFG, shard_tokens=400, n_workers=1)
    expected = sum(len(byte_tok.encode(d)) + 1 for d in source)  # +1 for EOS... byte has none
    if meta.eos_between_docs:
        assert meta.splits["train"].n_tokens == expected
    else:
        assert meta.splits["train"].n_tokens == expected - len(source)


# ---------------------------------------------------------------- dataset


def test_targets_are_inputs_shifted_by_one(packed):
    """The bug that makes loss look *better*: unshifted targets leak the answer."""
    ds = PackedDataset(packed, seq_len=32)
    x, y = ds.get_batch(8, step=0)
    assert x.shape == y.shape == (8, 32)
    assert (x[:, 1:] == y[:, :-1]).all()


def test_batches_are_a_function_of_seed_and_step(packed):
    ds = PackedDataset(packed, seq_len=16)
    a, _ = ds.get_batch(4, step=11)
    b, _ = ds.get_batch(4, step=11)
    c, _ = ds.get_batch(4, step=12)
    d, _ = ds.get_batch(4, step=11, seed=1)

    assert (a == b).all(), "same (seed, step) must replay exactly"
    assert not (a == c).all(), "different steps must differ"
    assert not (a == d).all(), "different seeds must differ"


def test_call_order_does_not_affect_a_step(packed):
    """Resuming at step N must see what an uninterrupted run saw at step N."""
    fresh = PackedDataset(packed, seq_len=16).get_batch(4, step=50)[0]

    ds = PackedDataset(packed, seq_len=16)
    for s in range(50):
        ds.get_batch(4, step=s)
    resumed = ds.get_batch(4, step=50)[0]

    assert (fresh == resumed).all()


def test_windows_never_straddle_a_shard(packed):
    ds = PackedDataset(packed, seq_len=16)
    sizes = [s.n_tokens for s in ds.info.shards]
    assert len(sizes) > 1, "fixture should be multi-shard"

    rng = np.random.default_rng(0)
    flat = rng.integers(0, ds.n_valid_positions, size=2000)
    shard_idx, offset = ds._locate(flat)
    for si, off in zip(shard_idx.tolist(), offset.tolist(), strict=True):
        assert off + ds.seq_len + 1 <= sizes[si]


def test_all_ids_are_within_vocab(packed):
    ds = PackedDataset(packed, seq_len=16)
    x, y = ds.get_batch(32, step=3)
    assert int(x.max()) < ds.vocab_size
    assert int(y.max()) < ds.vocab_size
    assert int(x.min()) >= 0


def test_stale_shards_are_refused(packed):
    """The failure this whole design exists to prevent."""
    other = build_tokenizer(GPT2_CFG)
    with pytest.raises(ValueError, match="packed with"):
        PackedDataset(packed, seq_len=16, tokenizer=other)


def test_matching_tokenizer_is_accepted(packed, byte_tok):
    PackedDataset(packed, seq_len=16, tokenizer=byte_tok)


def test_truncated_shard_is_detected(packed):
    meta = PackMeta.load(packed / "meta.json")
    victim = packed / meta.splits["train"].shards[0].file
    victim.write_bytes(victim.read_bytes()[:-8])
    with pytest.raises(ValueError, match="truncated|dtype"):
        PackedDataset(packed, seq_len=16)


def test_unknown_split_lists_what_exists(packed):
    with pytest.raises(KeyError, match="train"):
        PackedDataset(packed, seq_len=16, split="nope")


def test_seq_len_larger_than_any_shard_is_refused(packed):
    with pytest.raises(ValueError, match="no shard"):
        PackedDataset(packed, seq_len=100_000)


def test_uint32_path_end_to_end(tmp_path):
    """A vocab above 65,535 must survive the round trip byte-for-byte."""
    tok = build_tokenizer({"type": "tiktoken", "encoding": "o200k_base"})
    cfg = {"type": "tiktoken", "encoding": "o200k_base"}
    text = "研究人员发表了关于分词效率的研究成果。" * 40
    pack([text], tok, tmp_path, tokenizer_cfg=cfg, n_workers=1)

    ds = PackedDataset(tmp_path, seq_len=8, tokenizer=tok)
    assert ds.dtype == np.uint32
    ids = ds.read(0, len(ds)).tolist()
    assert max(ids) > 65_535, "test is pointless unless it exercises the wide range"
    assert tok.decode([i for i in ids if i != tok.eos_id]) == text


# ---------------------------------------------------------------- sources


def test_text_source():
    assert list(from_text("one")) == ["one"]
    assert list(from_text(["a", "b"])) == ["a", "b"]


def test_files_source(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    assert list(from_files("*.txt", root=tmp_path)) == ["alpha", "beta"]


def test_files_source_per_line(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\n\nthree\n")
    assert list(from_files("*.txt", root=tmp_path, per_line=True)) == ["one", "two", "three"]


def test_jsonl_source(tmp_path):
    (tmp_path / "d.jsonl").write_text('{"text": "first"}\n{"text": "second"}\n')
    assert list(from_jsonl("*.jsonl", root=tmp_path)) == ["first", "second"]


def test_build_source_from_config():
    assert list(build_source({"type": "text", "text": ["x", "y"]})) == ["x", "y"]


def test_unknown_source_names_alternatives():
    with pytest.raises(KeyError, match="files"):
        build_source({"type": "nope"})


def test_sources_stay_lazy():
    """A corpus must never be materialised — pack() relies on this."""
    import types

    assert isinstance(from_text(["a"]), types.GeneratorType)
