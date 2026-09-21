"""Data configs, the holdout split, and prepare()'s three cases.

Everything here runs offline: local text and the byte tokenizer. The one Hub
behaviour worth pinning — the ``subset`` argument — is checked against a fake
``datasets`` module rather than the network.
"""

import json
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from litterbox.data import PackedDataset, holdout, is_prepared, load_data_config, prepare
from litterbox.data.prepare import SIDECAR, DataConfig, main
from litterbox.data.source import build_source

CONFIG_DIR = Path(__file__).parent.parent / "configs" / "data"
BYTE = {"type": "byte"}


def docs(n, prefix="document"):
    return [f"{prefix} number {i}: " + "lorem ipsum dolor sit amet " * 6 for i in range(n)]


def explicit(tmp_path, **extra):
    return DataConfig(
        out_dir=str(tmp_path / "packed"),
        tokenizer=BYTE,
        splits={
            "train": {"type": "text", "text": docs(40)},
            "val": {"type": "text", "text": docs(8, "held")},
        },
        **extra,
    )


# ------------------------------------------------------------------ holdout


def test_holdout_partitions_every_document_exactly_once():
    corpus = docs(5000)
    train = list(holdout(corpus, val_fraction=0.1, keep="train"))
    val = list(holdout(corpus, val_fraction=0.1, keep="val"))
    assert not set(train) & set(val)
    assert sorted(train + val) == sorted(corpus)
    assert 0.07 < len(val) / len(corpus) < 0.13  # binomial around 10%, not exact


def test_holdout_is_a_function_of_the_text_not_of_order_or_history():
    """The property that makes it reproducible anywhere: a document's side
    does not depend on what came before it or on where it sits in the corpus."""
    corpus = docs(2000)
    first = set(holdout(corpus, val_fraction=0.1, keep="val"))
    assert set(holdout(reversed(corpus), val_fraction=0.1, keep="val")) == first
    assert set(holdout(corpus[500:], val_fraction=0.1, keep="val")) == first & set(corpus[500:])


def test_holdout_sends_exact_duplicates_to_the_same_side():
    """Web text repeats. Two copies of a page must not end up one in train and
    one in validation, or the validation loss measures memorisation."""
    corpus = docs(300) * 3
    train = set(holdout(corpus, val_fraction=0.2, keep="train"))
    val = set(holdout(corpus, val_fraction=0.2, keep="val"))
    assert not train & val


def test_holdout_seed_changes_the_partition_and_bad_arguments_raise():
    corpus = docs(2000)
    a = set(holdout(corpus, val_fraction=0.1, keep="val", seed=0))
    b = set(holdout(corpus, val_fraction=0.1, keep="val", seed=1))
    assert a != b
    with pytest.raises(ValueError, match="keep"):
        list(holdout(corpus, val_fraction=0.1, keep="test"))
    with pytest.raises(ValueError, match="val_fraction"):
        list(holdout(corpus, val_fraction=1.0, keep="val"))


# ------------------------------------------------------------------- schema


def test_shipped_data_configs_all_validate():
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(paths) >= 3
    for path in paths:
        cfg = load_data_config(path)
        assert cfg.out_dir.startswith("data/"), path.name  # the gitignored top-level folder


def test_config_is_strict_about_unknown_keys():
    with pytest.raises(ValidationError, match="tokeniser"):
        DataConfig(out_dir="x", tokeniser=BYTE, splits={"train": {"type": "text", "text": "a"}})
    with pytest.raises(ValidationError, match="val_frac"):
        DataConfig(out_dir="x", tokenizer=BYTE, source={"type": "text"}, holdout={"val_frac": 0.1})


def test_config_demands_exactly_one_shape():
    src = {"type": "text", "text": "a"}
    with pytest.raises(ValidationError, match="exactly one"):
        DataConfig(out_dir="x", tokenizer=BYTE)
    with pytest.raises(ValidationError, match="exactly one"):
        DataConfig(out_dir="x", tokenizer=BYTE, source=src, splits={"train": src})
    with pytest.raises(ValidationError, match="needs a `holdout`"):
        DataConfig(out_dir="x", tokenizer=BYTE, source=src)
    with pytest.raises(ValidationError, match="must include 'train'"):
        DataConfig(out_dir="x", tokenizer=BYTE, splits={"val": src})


def test_fingerprint_tracks_content_not_location():
    base = DataConfig(out_dir="a", tokenizer=BYTE, splits={"train": {"type": "text", "text": "x"}})
    moved = base.model_copy(update={"out_dir": "b", "description": "reworded"})
    limited = base.model_copy(update={"limit": 10})
    assert moved.fingerprint() == base.fingerprint()
    assert limited.fingerprint() != base.fingerprint()


# ------------------------------------------------------------------ prepare


def test_prepare_packs_every_split_and_training_can_read_it(tmp_path):
    cfg = explicit(tmp_path)
    meta = prepare(cfg, progress=False)

    assert set(meta.splits) == {"train", "val"}
    assert meta.splits["train"].n_documents == 40 and meta.splits["val"].n_documents == 8
    sidecar = json.loads((Path(cfg.out_dir) / SIDECAR).read_text())
    assert sidecar["fingerprint"] == cfg.fingerprint()

    x, y = PackedDataset(cfg.out_dir, seq_len=32, split="val").get_batch(4, step=0)
    assert x.shape == (4, 32) and (x[:, 1:] == y[:, :-1]).all()


def test_prepare_from_one_source_carves_out_validation(tmp_path):
    corpus = docs(400)
    cfg = DataConfig(
        out_dir=str(tmp_path / "packed"),
        tokenizer=BYTE,
        source={"type": "text", "text": corpus},
        holdout={"val_fraction": 0.1},
    )
    meta = prepare(cfg, progress=False)
    n_train, n_val = meta.splits["train"].n_documents, meta.splits["val"].n_documents
    assert n_train + n_val == 400 and 20 < n_val < 60


def test_prepare_is_a_no_op_when_the_directory_already_matches(tmp_path, monkeypatch):
    cfg = explicit(tmp_path)
    prepare(cfg, progress=False)
    assert is_prepared(cfg)

    def boom(*a, **k):
        raise AssertionError("pack() must not run again for an up-to-date directory")

    # By module object, not dotted string: the package re-exports a `prepare`
    # function, which shadows the `prepare` module under attribute lookup.
    monkeypatch.setattr(sys.modules["litterbox.data.prepare"], "pack", boom)
    prepare(cfg, progress=False)


def test_prepare_refuses_a_directory_packed_from_another_config(tmp_path):
    prepare(explicit(tmp_path), progress=False)
    changed = explicit(tmp_path, limit=5)
    assert not is_prepared(changed)
    with pytest.raises(FileExistsError, match="different config"):
        prepare(changed, progress=False)

    meta = prepare(changed, force=True, progress=False)
    assert meta.splits["train"].n_documents == 5 and is_prepared(changed)


def test_a_run_that_died_partway_does_not_read_as_up_to_date(tmp_path):
    cfg = explicit(tmp_path)
    prepare(cfg, progress=False)
    (Path(cfg.out_dir) / SIDECAR).unlink()  # the sidecar is written last
    assert not is_prepared(cfg)
    with pytest.raises(FileExistsError, match="died partway"):
        prepare(cfg, progress=False)


def test_force_will_not_delete_a_directory_that_is_not_a_dataset(tmp_path):
    """A typo in out_dir plus --force must not be able to remove real files."""
    precious = tmp_path / "packed"
    precious.mkdir()
    (precious / "thesis.tex").write_text("irreplaceable")
    with pytest.raises(FileExistsError, match="does not look like a packed dataset"):
        prepare(explicit(tmp_path), force=True, progress=False)
    assert (precious / "thesis.tex").read_text() == "irreplaceable"


def test_cli_round_trip(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i, text in enumerate(docs(60)):
        (corpus / f"{i:03d}.txt").write_text(text)
    cfg_path = tmp_path / "data.yaml"
    cfg_path.write_text(
        f"out_dir: {tmp_path / 'out'}\n"
        "tokenizer: {type: byte}\n"
        f"source: {{type: files, pattern: '*.txt', root: {corpus}}}\n"
        "holdout: {val_fraction: 0.2}\n"
    )
    assert main([str(cfg_path)]) == 0
    assert "tokens as uint16" in capsys.readouterr().out
    assert main([str(cfg_path)]) == 0
    assert "up to date" in capsys.readouterr().out
    assert main([str(cfg_path), "--limit", "3"]) == 1  # different fingerprint, no --force
    assert "--force" in capsys.readouterr().err


# --------------------------------------------------------------- hub source


def test_huggingface_source_passes_subset_as_the_config_name(monkeypatch):
    """The Hub API calls a dataset's configuration ``name``, which collides
    with this source's own ``name`` (the repo). ``subset`` is the way through,
    and FineWeb-Edu's sample cannot be selected without it."""
    seen = {}

    def fake_load_dataset(path, config=None, **kwargs):
        seen.update(path=path, config=config, **kwargs)
        return [{"text": "one"}, {"text": "two"}, {"text": "three"}]

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset)
    )
    out = list(
        build_source(
            {"type": "huggingface", "name": "org/corpus", "subset": "sample-10BT", "limit": 2}
        )
    )
    assert out == ["one", "two"]
    assert seen["path"] == "org/corpus" and seen["config"] == "sample-10BT"
    assert seen["split"] == "train" and seen["streaming"] is True
