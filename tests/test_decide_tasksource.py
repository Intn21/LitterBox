"""The tasksource adapter: public rows in, litterbox records out, filters applied.

The rows below are real ones from tasksource-jev-typed-decisions, shortened.
No test touches the network; ``build`` is given its rows directly.
"""

import json
from collections import Counter

import pytest

from litterbox.decide.data.record import read_jsonl
from litterbox.decide.data.tasksource import (
    JEV_BENCH_SOURCES,
    Skip,
    build,
    convert,
    in_slice,
    to_record,
)
from litterbox.decide.readout import render
from litterbox.decide.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion


def row(**kw):
    base = {
        "state": "I work at a small grocery store behind the deli counter.",
        "kind": "choice",
        "id": "scruples:train:1",
        "options": ["the author is in the wrong", "the other party is in the wrong", "no one is"],
        "target": [0.0, 1.0, 0.0],
        "question": "Who is in the wrong in this story?",
        "source": "scruples/verdict",
        "variant": "direct",
        "split": "train",
        "group_id": "scruples:train:1",
        "question_id": "decision",
        "license": "apache-2.0",
        "license_use": "commercial",
    }
    return {**base, **kw}


SCORE = row(
    kind="score",
    id="sick:1",
    options=["1 (unrelated)", "2", "3", "4", "5 (related)"],
    target=[0.8, 0.2, 0.0, 0.0, 0.0],
    source="sick/relatedness",
    question_id="sick",
)
NOUL = row(
    kind="noul",
    id="lonli:1",
    options=[],
    target=[1.0],
    source="lonli",
    question="Does text_A entail text_B?",
    question_id="noul-label-verification",
)


# --- one row ------------------------------------------------------------------------


def test_a_one_hot_choice_becomes_a_label_and_no_truth():
    r = to_record(row())
    q = r.questions["decision"]
    assert isinstance(q, ChoiceQuestion) and list(q.criteria) == row()["options"]
    assert r.labels == {"decision": 1} and r.truth is None and r.region is None
    assert r.source == "tasksource/scruples/verdict"
    assert r.group == "scruples:train:1" and r.license == "commercial"


def test_a_soft_target_becomes_a_truth_with_the_likeliest_answer_as_label():
    r = to_record(SCORE)
    assert isinstance(r.questions["sick"], ScoreQuestion)
    assert r.truth == {"sick": pytest.approx([0.8, 0.2, 0, 0, 0])} and r.labels == {"sick": 0}


def test_a_noul_is_reordered_from_p_yes_to_no_yes():
    r = to_record(NOUL)
    assert isinstance(r.questions["noul-label-verification"], NoulQuestion)
    assert r.labels == {"noul-label-verification": 1} and r.truth is None
    soft = to_record({**NOUL, "target": [0.25]})
    assert soft.truth == {"noul-label-verification": pytest.approx([0.75, 0.25])}
    assert soft.labels == {"noul-label-verification": 0}


def test_a_scale_longer_than_ten_levels_becomes_a_choice():
    levels = [f"{i} of 12" for i in range(12)]
    r = to_record(row(kind="score", options=levels, target=[1.0] + [0.0] * 11))
    assert isinstance(r.questions["decision"], ChoiceQuestion)


def test_converted_records_render_through_the_readout():
    for source in (row(), SCORE, NOUL):
        r = to_record(source)
        assert render(r.request(), next(iter(r.questions))).endswith("Answer:")


@pytest.mark.parametrize(
    "bad, reason",
    [
        (row(target=[0.5, 0.2, 0.1]), "bad_target"),
        (row(target=[0.0, 1.0]), "bad_target"),
        (row(target=[-0.1, 1.1, 0.0]), "bad_target"),
        ({**NOUL, "target": [1.5]}, "bad_target"),
        ({**NOUL, "target": [0.2, 0.8]}, "bad_target"),
        (row(options=["a", "a", "b"]), "duplicate_or_empty_options"),
        (row(options=["a", "", "b"]), "duplicate_or_empty_options"),
        (
            row(options=[f"o{i}" for i in range(256)], target=[1.0] + [0.0] * 255),
            "too_many_options",
        ),
        (row(options=["only"], target=[1.0]), "invalid"),
        (row(state=7), "invalid"),
    ],
)
def test_rows_that_cannot_become_records_are_skipped_with_a_reason(bad, reason):
    with pytest.raises(Skip, match=reason):
        to_record(bad)


# --- filters ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    ["PolyAI/banking77", "super_glue/boolq", "glue/mnli", "chaos-mnli-ambiguity/votes", "sst2",
     "glue/sst2", "HelpSteer2/coherence", "clinc_oos/plus", "ai2_arc/ARC-Challenge", "fever/v1"],
)  # fmt: skip
def test_jev_bench_sources_are_excluded(source):
    stats = Counter()
    assert list(convert([row(source=source)], stats=stats)) == []
    assert stats["skipped/excluded_source"] == 1 and stats[f"excluded/{source}"] == 1


@pytest.mark.parametrize(
    "source", ["scruples/verdict", "WANLI", "HelpSteer/coherence", "dynasent/r1", "esconv/strategy"]
)
def test_other_sources_are_kept(source):
    assert len(list(convert([row(source=source)]))) == 1


def test_exclusion_can_be_turned_off_and_licenses_filtered():
    rows = [row(source="boolq"), row(license_use="non-commercial"), row(license_use="unspecified")]
    assert len(list(convert(rows, exclude=()))) == 3
    stats = Counter()
    kept = list(convert(rows, exclude=(), licenses=["commercial"], stats=stats))
    assert len(kept) == 1 and stats["skipped/license"] == 2 and stats["kept"] == 1
    assert len(JEV_BENCH_SOURCES) == 19


def test_bad_rows_are_counted_not_raised():
    stats = Counter()
    kept = list(convert([row(), row(target=[1, 1, 1]), row(options=["a", "a", "b"])], stats=stats))
    assert len(kept) == 1
    assert stats["skipped/bad_target"] == 1 and stats["skipped/duplicate_or_empty_options"] == 1


# --- splitting and files --------------------------------------------------------------


def test_in_slice_is_stable_and_close_to_its_share():
    groups = [f"g{i}" for i in range(20000)]
    picked = [g for g in groups if in_slice(g, 0.05)]
    assert len(picked) / 20000 == pytest.approx(0.05, abs=0.006)
    assert picked == [g for g in groups if in_slice(g, 0.05)]
    assert picked != [g for g in groups if in_slice(g, 0.05, seed=1)]


def test_build_keeps_groups_whole_and_writes_a_manifest(tmp_path):
    # 400 groups of two rows each: two questions about the same state.
    train = []
    for g in range(400):
        for j in range(2):
            train.append(row(id=f"t{g}:{j}", group_id=f"g{g}", question_id=f"q{j}"))
    train += [row(id="x", source="boolq"), row(id="y", target=[1, 1, 1])]
    rows = {
        "train": train,
        "validation": [row(id="v0", group_id="vg0")],
        "test": [row(id="e0", group_id="eg0"), SCORE],
    }
    manifest = build(tmp_path, calibration_share=0.25, rows_by_split=rows, revision="abc123")

    split = {name: list(read_jsonl(tmp_path / f"{name}.jsonl")) for name in manifest["splits"]}
    assert len(split["train"]) + len(split["calibration"]) == 800
    assert 120 < len(split["calibration"]) < 280
    assert {r.group for r in split["train"]}.isdisjoint({r.group for r in split["calibration"]})
    assert all(v == 0 for v in manifest["shared_groups"].values())

    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["revision"] == "abc123"
    assert on_disk["skipped"] == {"excluded_source": 1, "bad_target": 1}
    assert on_disk["excluded_sources"] == {"boolq": 1}
    assert on_disk["splits"]["test"]["soft_targets"] == 1
    assert on_disk["splits"]["test"]["kinds"] == {"choice": 1, "score": 1}
    assert on_disk["splits"]["train"]["groups"] * 2 == on_disk["splits"]["train"]["records"]


def test_build_reports_groups_shared_between_splits(tmp_path):
    rows = {"train": [row(id="a", group_id="shared")], "test": [row(id="b", group_id="shared")]}
    manifest = build(tmp_path, calibration_share=0.0, rows_by_split=rows)
    assert manifest["shared_groups"]["train&test"] == 1
