"""The generated corpora: records whose label frequencies were written down.

The claims worth pinning are the ones evaluation will lean on: the drawn
labels really follow each region's die, the recorded truth is the honest
answer given the state, nothing about a hidden cue reaches an unrecoverable
state, and a corpus is reproducible from its seed.
"""

import json
from collections import Counter, defaultdict

import pytest
from pydantic import ValidationError

from litterbox.decide.calibration import ece, ece_noise_floor
from litterbox.decide.data import DOMAINS, Domain, QuestionSpec, Record, Region, generate
from litterbox.decide.data.cli import build
from litterbox.decide.data.record import read_jsonl, write_jsonl
from litterbox.decide.data.regions import surfaces
from litterbox.decide.readout import render

N = 6000


@pytest.fixture(scope="module", params=sorted(DOMAINS))
def corpus(request):
    return DOMAINS[request.param], generate(DOMAINS[request.param], N, seed=0)


def canonical(spec, record):
    """The drawn answer as an index into the spec's own option order."""
    label = record.labels[spec.qid]
    if spec.type != "choice":
        return label
    return spec.options.index(list(record.questions[spec.qid].criteria)[label])


# --- every domain -------------------------------------------------------------------


def test_shipped_domains_are_well_formed():
    assert set(DOMAINS) == {"tickets", "logs", "orders"}
    for domain in DOMAINS.values():
        domain.check()
        assert domain.hide, "every domain names a question to hide in the unrecoverable corpus"


def test_records_validate_and_become_requests_the_readout_can_render(corpus):
    _, records = corpus
    for r in records[:50]:
        req = r.request()
        for qid in req.questions:
            assert render(req, qid).endswith("Answer:")


def test_drawn_labels_follow_each_regions_die(corpus):
    domain, records = corpus
    for spec in domain.questions:
        by_region = defaultdict(list)
        for r in records:
            by_region[r.region[spec.qid]].append(canonical(spec, r))
        assert len(by_region) == len(spec.regions)
        for region in spec.regions:
            drawn = by_region[f"{region.kind}/{region.name}"]
            n = len(drawn)
            assert n / N == pytest.approx(region.weight, abs=0.03)
            for j, p in enumerate(region.dist):
                tolerance = 4 * (p * (1 - p) / n) ** 0.5 + 1e-9  # four standard errors
                assert drawn.count(j) / n == pytest.approx(p, abs=tolerance), (
                    spec.qid,
                    region.name,
                )


def test_recorded_truth_is_the_regions_die_in_the_shown_order(corpus):
    domain, records = corpus
    dice = {q.qid: {f"{r.kind}/{r.name}": r.dist for r in q.regions} for q in domain.questions}
    for r in records[:500]:
        for spec in domain.questions:
            die = dice[spec.qid][r.region[spec.qid]]
            shown = r.truth[spec.qid]
            if spec.type == "choice":
                names = list(r.questions[spec.qid].criteria)
                assert [die[spec.options.index(name)] for name in names] == pytest.approx(shown)
            else:
                assert list(die) == pytest.approx(shown)


def test_truth_is_calibrated_against_the_labels(corpus):
    # The answer key itself, scored as if it were a model: its ECE must sit on the noise floor.
    domain, records = corpus
    for spec in domain.questions:
        p = [r.truth[spec.qid] for r in records]
        y = [r.labels[spec.qid] for r in records]
        floor, std = ece_noise_floor(p, reps=50)
        assert abs(ece(p, y) - floor) < 4 * std + 1e-9


def test_choice_options_are_shuffled_evenly_and_scores_are_not(corpus):
    domain, records = corpus
    for spec in domain.questions:
        if spec.type == "choice":
            first = Counter(list(r.questions[spec.qid].criteria)[0] for r in records)
            for name in spec.options:
                assert first[name] / N == pytest.approx(1 / len(spec.options), abs=0.03)
        elif spec.type == "score":
            assert all(r.questions[spec.qid].criteria == list(spec.options) for r in records[:200])


def test_the_same_seed_gives_the_same_corpus_and_another_seed_does_not(corpus):
    domain, records = corpus
    again = generate(domain, 200, seed=0)
    assert [r.model_dump() for r in again] == [r.model_dump() for r in records[:200]]
    assert generate(domain, 200, seed=1)[0].state != records[0].state


# --- the unrecoverable corpus -----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_hidden_cues_never_reach_an_unrecoverable_state(name):
    domain = DOMAINS[name]
    records = generate(domain, 2000, seed=3, unrecoverable=True)
    hidden = [q for q in domain.questions if q.qid in domain.hide]
    forms = [s for q in hidden for r in q.regions for c in r.cues for s in surfaces(c)]
    assert forms
    for r in records:
        text = r.state if isinstance(r.state, str) else None
        for form in forms:
            assert (form not in text) if text is not None else (form not in r.state)
        for q in hidden:
            assert r.region[q.qid] == "unrecoverable/hidden"
            assert r.truth[q.qid] == pytest.approx(q.base_rate())
        assert r.source.endswith("-unrecoverable")


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_hidden_labels_still_follow_the_base_rate_and_other_questions_are_untouched(name):
    domain = DOMAINS[name]
    records = generate(domain, N, seed=3, unrecoverable=True)
    for spec in domain.questions:
        if spec.qid in domain.hide:
            counts = Counter(r.labels[spec.qid] for r in records)
            for j, p in enumerate(spec.base_rate()):
                assert counts[j] / N == pytest.approx(p, abs=0.03)
        else:
            assert all(r.kind(spec.qid) in ("deterministic", "stochastic") for r in records)


def test_the_normal_corpus_does_plant_the_cues_the_unrecoverable_one_hides():
    domain = DOMAINS["tickets"]
    spec = next(q for q in domain.questions if q.qid == "urgency")
    forms = [c for r in spec.regions for c in r.cues]
    with_cue = sum(any(f in r.state for f in forms) for r in generate(domain, 1000, seed=0))
    assert with_cue / 1000 == pytest.approx(1 - 0.25, abs=0.05)  # all but the 'unstated' region


# --- specs that must be refused -----------------------------------------------------------

Q = dict(type="noul", instructions=("ok?",))


def domain_of(*questions, filler=()):
    return Domain("d", questions, lambda rng, planted: " ".join(planted.values()), filler=filler)


def test_a_cue_shared_between_questions_is_refused():
    a = QuestionSpec(
        "a", regions=(Region("x", "deterministic", (0, 1), 1.0, ("please hurry",)),), **Q
    )
    b = QuestionSpec("b", regions=(Region("y", "deterministic", (0, 1), 1.0, ("hurry",)),), **Q)
    with pytest.raises(ValueError, match="share the surface"):
        domain_of(a, b).check()
    with pytest.raises(ValueError, match="share the surface"):
        domain_of(a, filler=("so please hurry up",)).check()


@pytest.mark.parametrize(
    "regions, message",
    [
        ((Region("x", "deterministic", (0, 1), 0.5, ("a",)),), "weights must sum to 1"),
        ((Region("x", "deterministic", (0.5, 0.5), 1.0, ("a",)),), "does not match its dist"),
        ((Region("x", "stochastic", (0, 1), 1.0, ("a",)),), "does not match its dist"),
        ((Region("x", "stochastic", (0.5, 0.6), 1.0, ("a",)),), "not a distribution"),
        ((Region("x", "deterministic", (0, 1, 0), 1.0, ("a",)),), "3 entries, not 2"),
        (
            (Region("x", "deterministic", (0, 1), 0.5), Region("y", "deterministic", (1, 0), 0.5)),
            "at most one region may have no cue",
        ),
        (
            (
                Region("x", "deterministic", (0, 1), 0.5, ("same words",)),
                Region("y", "deterministic", (1, 0), 0.5, ("the same words again",)),
            ),
            "cannot be told apart",
        ),
    ],
)
def test_malformed_question_specs_are_refused(regions, message):
    with pytest.raises(ValueError, match=message):
        QuestionSpec("q", regions=regions, **Q).check()


def test_record_refuses_labels_and_truth_that_do_not_fit_its_questions():
    base = dict(
        id="r", source="s", state="x", questions={"q": {"type": "noul", "instructions": "?"}}
    )
    Record(**base, labels={"q": 1}, truth={"q": [0.2, 0.8]}, region={"q": "stochastic/x"})
    Record(**base, labels={"q": 0})  # public data: no truth, no region
    for bad in (
        dict(labels={"q": 2}),
        dict(labels={}),
        dict(labels={"q": 1}, truth={"q": [0.2, 0.7]}),
        dict(labels={"q": 1}, truth={"q": [0.2, 0.3, 0.5]}),
        dict(labels={"q": 1}, region={"other": "x"}),
    ):
        with pytest.raises(ValidationError):
            Record(**base, **bad)


# --- files ------------------------------------------------------------------------------


def test_jsonl_round_trip(tmp_path):
    records = generate(DOMAINS["orders"], 50, seed=0)
    assert write_jsonl(tmp_path / "x.jsonl", records) == 50
    assert list(read_jsonl(tmp_path / "x.jsonl")) == records


def test_build_writes_three_splits_and_a_manifest(tmp_path):
    manifest = build("tickets", 1000, 7, False, tmp_path)
    assert manifest["splits"] == {"train": 800, "calibration": 100, "test": 100}
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["seed"] == 7 and on_disk["unrecoverable"] is False
    dept = on_disk["questions"]["department"]
    assert sum(dept["records_per_region"].values()) == 1000
    assert {r["name"]: r["dist"] for r in dept["regions"]}["charged_twice"] == [0.6, 0, 0.4, 0]
    ids = [
        r.id
        for split in ("train", "calibration", "test")
        for r in read_jsonl(tmp_path / f"{split}.jsonl")
    ]
    assert len(set(ids)) == 1000

    hidden = build("tickets", 200, 7, True, tmp_path / "u")
    assert hidden["hidden_questions"] == ["urgency"]
    assert hidden["questions"]["urgency"]["records_per_region"] == {"unrecoverable/hidden": 200}
