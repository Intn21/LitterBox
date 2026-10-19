"""Calibration measured on groups: proper scores, ECE and its floor, coverage.

The load-bearing test is the one the roadmap's step 0 names: predictions that
are honest by construction, labels drawn from the probabilities themselves,
must measure an ECE on the noise floor. That proves the module measures the
gap and not its own binning.
"""

import json

import pytest
import torch

from litterbox.decide.calibration import (
    Bin,
    accuracy,
    as_probabilities,
    aurc,
    brier,
    coverage_at,
    ece,
    ece_noise_floor,
    log_loss,
    reliability,
    reliability_diagram,
    risk_coverage,
    summarize,
)


def honest(n, k, seed=0, sharpness=2.0):
    """Random distributions and labels drawn from them: calibrated by construction."""
    g = torch.Generator().manual_seed(seed)
    p = (torch.randn(n, k, generator=g) * sharpness).softmax(-1)
    y = torch.multinomial(p, 1, generator=g).squeeze(1)
    return p, y


# --- per-answer scores ---------------------------------------------------------------


def test_brier_and_log_loss_by_hand():
    p = [[0.81, 0.12, 0.07], [0.2, 0.8]]
    y = [0, 0]
    assert brier(p, y) == pytest.approx(((0.19**2 + 0.12**2 + 0.07**2) + (0.8**2 + 0.8**2)) / 2)
    assert log_loss(p, y) == pytest.approx(
        (-torch.log(torch.tensor(0.81)) - torch.log(torch.tensor(0.2))).item() / 2
    )
    assert accuracy(p, y) == 0.5


def test_ragged_rows_are_padded_with_zeros():
    p = as_probabilities([[0.5, 0.5], [0.2, 0.3, 0.5]])
    assert p.shape == (2, 3) and p[0, 2] == 0.0


# --- ECE ------------------------------------------------------------------------------


def test_ece_matches_the_two_bin_hand_example():
    said = [0.6] * 8 + [0.9] * 12
    right = [1, 1, 1, 1, 0, 0, 0, 0] + [1] * 11 + [0]
    p = [[s, 1 - s] for s in said]
    y = [0 if r else 1 for r in right]  # label 0 when the confident option was right
    assert ece(p, y, bins=4) == pytest.approx(
        (8 / 20) * 0.10 + (12 / 20) * abs(0.9 - 11 / 12), abs=1e-6
    )
    rows = reliability(p, y, bins=4)
    assert [b.count for b in rows] == [8, 12]
    assert rows[0].gap == pytest.approx(0.10)


def test_a_confidence_of_exactly_zero_or_one_lands_in_a_bin():
    p = [[1.0, 0.0], [0.0, 1.0]]
    assert sum(b.count for b in reliability(p, [0, 1])) == 2


def test_equal_mass_bins_hold_equal_counts_where_width_bins_do_not():
    p, y = honest(1000, 4, sharpness=4.0)  # mostly very confident: width bins pile up at the top
    width = reliability(p, y, bins=5, scheme="width")
    mass = reliability(p, y, bins=5, scheme="mass")
    assert max(b.count for b in width) > 500
    assert max(b.count for b in mass) <= 210 and len(mass) == 5


def test_honest_predictions_measure_an_ece_on_the_noise_floor():
    # The step 0 exit condition. Labels drawn from the probabilities themselves:
    # whatever ECE remains is binning noise, and the floor estimates exactly that.
    for k, scheme in [(2, "width"), (4, "width"), (4, "mass"), (7, "mass")]:
        p, y = honest(2000, k, seed=k)
        measured = ece(p, y, scheme=scheme)
        floor, std = ece_noise_floor(p, scheme=scheme, reps=100, seed=1)
        assert abs(measured - floor) < 3 * std + 1e-9, (k, scheme, measured, floor, std)


def test_overconfident_predictions_sit_far_above_the_floor():
    p, y = honest(2000, 4)
    sharpened = (p.log() * 3).softmax(-1)  # same answers, much surer
    floor, std = ece_noise_floor(sharpened, reps=50)
    assert ece(sharpened, y) > floor + 10 * std
    assert accuracy(sharpened, y) == accuracy(p, y)  # accuracy did not move; honesty did


def test_calibration_is_independent_of_accuracy():
    # Fifty-fifty on every answer, right half the time: useless, and perfectly honest.
    p = [[0.5, 0.5]] * 100
    y = [0, 1] * 50
    assert accuracy(p, y) == 0.5
    assert ece(p, y) == pytest.approx(0.0, abs=1e-6)


def test_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match="scheme"):
        reliability([[1.0, 0.0]], [0], scheme="quantile")


# --- ordering -------------------------------------------------------------------------


def test_coverage_counts_from_the_top_of_the_confidence_order():
    p = [[0.99, 0.01], [0.9, 0.1], [0.8, 0.2], [0.6, 0.4]]
    y = [0, 0, 1, 0]  # the third most confident is wrong
    cov, risk = risk_coverage(p, y)
    assert cov.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert risk.tolist() == pytest.approx([0.0, 0.0, 1 / 3, 0.25])
    assert coverage_at(p, y, budget=0.05) == 0.5
    assert coverage_at(p, y, budget=0.30) == 1.0
    assert aurc(p, y) == pytest.approx((0 + 0 + 1 / 3 + 0.25) / 4)


def test_coverage_is_zero_when_the_most_confident_answer_is_wrong_and_one_when_all_are_right():
    assert coverage_at([[0.9, 0.1], [0.6, 0.4]], [1, 0]) == 0.0
    assert coverage_at([[0.9, 0.1], [0.6, 0.4]], [0, 0]) == 1.0


def test_temperature_cannot_change_coverage_on_two_option_questions():
    p, y = honest(500, 2)
    logits = p.log()
    for T in (0.5, 3.0):
        assert coverage_at((logits / T).softmax(-1), y) == coverage_at(p, y)
        assert aurc((logits / T).softmax(-1), y) == pytest.approx(aurc(p, y))


def test_shuffling_confidences_destroys_coverage_but_not_accuracy():
    p, y = honest(1000, 3, sharpness=3.0)
    g = torch.Generator().manual_seed(0)
    conf = p.max(-1).values[torch.randperm(1000, generator=g)]
    top = torch.nn.functional.one_hot(p.argmax(-1), 3).float()
    shuffled = top * conf[:, None] + (1 - top) * ((1 - conf) / 2)[:, None]
    assert accuracy(shuffled, y) == accuracy(p, y)
    assert coverage_at(shuffled, y) < coverage_at(p, y) / 4


# --- summary and picture ---------------------------------------------------------------


def test_summary_is_json_ready_and_consistent_with_the_parts():
    p, y = honest(300, 3)
    s = summarize(p, y, reps=20)
    d = s.as_dict()
    json.dumps(d)
    assert d["n"] == 300
    assert d["ece_width"] == pytest.approx(ece(p, y))
    assert d["coverage_at_5"] == pytest.approx(coverage_at(p, y))
    assert isinstance(reliability(p, y)[0], Bin)


def test_reliability_diagram_draws_when_matplotlib_is_present():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    p, y = honest(200, 3)
    ax = reliability_diagram(p, y, title="honest")
    assert "ECE" in ax.get_title()
