"""Order records: structured state, a JSON object with its keys in shuffled order.

Each question reads one field; the rest are noise. A gift-card payment is
reviewed half the time, declined 30% and approved 20%. Account age is a
number, which decision models are known to read badly. In an unrecoverable
corpus the shipping field is absent, as if risk followed a merchant category
the record omits.
"""

from __future__ import annotations

from litterbox.decide.data.regions import Domain, QuestionSpec, Region

NOISE_FIELDS = ("order_id", "amount", "currency", "items", "country")

ACTION = QuestionSpec(
    qid="action",
    type="choice",
    instructions=("What should happen to this order?", "Choose the next action for this order."),
    options=("approve", "review", "decline", "hold"),
    regions=(
        Region(
            "saved_card", "deterministic", (1, 0, 0, 0), 0.20, ({"payment_method": "saved_card"},)
        ),
        Region("new_card", "deterministic", (0, 1, 0, 0), 0.20, ({"payment_method": "new_card"},)),
        Region(
            "blocked_card",
            "deterministic",
            (0, 0, 1, 0),
            0.20,
            ({"payment_method": "blocked_card"},),
        ),
        Region(
            "bank_transfer",
            "deterministic",
            (0, 0, 0, 1),
            0.20,
            ({"payment_method": "bank_transfer"},),
        ),
        Region(
            "gift_card", "stochastic", (0.2, 0.5, 0.3, 0), 0.20, ({"payment_method": "gift_card"},)
        ),
    ),
)

RISK = QuestionSpec(
    qid="risk",
    type="score",
    instructions=("How risky is this order?", "Rate the risk of this order."),
    options=("low", "medium", "high"),
    regions=(
        Region(
            "same_address", "deterministic", (1, 0, 0), 0.30, ({"shipping": "same_as_billing"},)
        ),
        Region("other_city", "deterministic", (0, 1, 0), 0.25, ({"shipping": "different_city"},)),
        Region("forwarder", "deterministic", (0, 0, 1), 0.20, ({"shipping": "freight_forwarder"},)),
        Region(
            "other_country", "stochastic", (0, 0.5, 0.5), 0.25, ({"shipping": "different_country"},)
        ),
    ),
)

FRAUD = QuestionSpec(
    qid="likely_fraud",
    type="noul",
    instructions=("Is this order likely fraudulent?",),
    regions=(
        Region(
            "new_account",
            "stochastic",
            (0.3, 0.7),
            0.25,
            tuple({"account_age_days": d} for d in (0, 1, 2)),
        ),
        Region(
            "old_account",
            "stochastic",
            (0.98, 0.02),
            0.75,
            tuple({"account_age_days": d} for d in (400, 900, 1500)),
        ),
    ),
)

QUESTIONS = (ACTION, RISK, FRAUD)


def render(rng, planted):
    state = {
        "order_id": f"ord-{rng.randrange(10**6):06d}",
        "amount": round(rng.uniform(5, 900), 2),
        "currency": rng.choice(("USD", "EUR", "BRL")),
        "items": rng.randint(1, 6),
        "country": rng.choice(("US", "DE", "BR", "JP", "FR")),
    }
    for cue in planted.values():
        state.update(cue)
    keys = list(state)
    rng.shuffle(keys)
    return {k: state[k] for k in keys}


DOMAIN = Domain("orders", QUESTIONS, render, filler=NOISE_FIELDS, hide=("risk",))
