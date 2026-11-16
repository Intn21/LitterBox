"""Support tickets: free text, assembled from sentences in shuffled order.

The stochastic regions are the point. "Charged twice" tickets are billing 60%
of the time and account 40%, with identical text either way, and a polite
"as soon as you can" is this-week 70% of the time and today 30%. In an
unrecoverable corpus the urgency cues are withheld, as if urgency followed a
customer tier the ticket never mentions.
"""

from __future__ import annotations

from litterbox.decide.data.regions import Domain, QuestionSpec, Region

GREETINGS = ("Hi,", "Hello,", "Hey team,", "Good morning,")
FILLER = (
    "Hope you are doing well.",
    "I have been a customer for a while.",
    "Thanks in advance.",
    "I am on the web version.",
    "Let me know if you need more details.",
    "I looked at the help pages first.",
)

DEPARTMENT = QuestionSpec(
    qid="department",
    type="choice",
    instructions=(
        "Which department should handle this ticket?",
        "Route this ticket to a department.",
        "Who should pick this up?",
    ),
    options=("billing", "technical", "account", "other"),
    regions=(
        Region(
            "billing_keyword",
            "deterministic",
            (1, 0, 0, 0),
            0.19,
            (
                "My invoice shows the wrong amount.",
                "I need a copy of last month's receipt.",
                "The payment page rejected my card.",
            ),
        ),
        Region(
            "technical_keyword",
            "deterministic",
            (0, 1, 0, 0),
            0.19,
            (
                "The app crashes when I open it.",
                "I get an error on the export page.",
                "The dashboard has become very slow.",
            ),
        ),
        Region(
            "account_keyword",
            "deterministic",
            (0, 0, 1, 0),
            0.19,
            (
                "I cannot log in.",
                "The password reset link does not work.",
                "My profile is locked.",
            ),
        ),
        Region(
            "other_keyword",
            "deterministic",
            (0, 0, 0, 1),
            0.19,
            (
                "I wanted to pass on some feedback about the new design.",
                "Where can I find your press kit?",
                "Do you have a public roadmap?",
            ),
        ),
        Region(
            "charged_twice",
            "stochastic",
            (0.6, 0, 0.4, 0),
            0.24,
            (
                "I was charged twice.",
                "I got billed 2x.",
                "There is a double charge on my statement.",
            ),
        ),
    ),
)

URGENCY = QuestionSpec(
    qid="urgency",
    type="score",
    instructions=("How urgent is this ticket?", "Rate the urgency of this request."),
    options=("can wait", "this week", "today"),
    regions=(
        Region(
            "today",
            "deterministic",
            (0, 0, 1),
            0.30,
            (
                "I need this fixed today.",
                "This is blocking us right now.",
            ),
        ),
        Region(
            "this_week",
            "deterministic",
            (0, 1, 0),
            0.30,
            (
                "Sometime in the next few days would be fine.",
                "No rush, but before Friday please.",
            ),
        ),
        Region("unstated", "deterministic", (1, 0, 0), 0.25),
        Region(
            "polite_soon",
            "stochastic",
            (0, 0.7, 0.3),
            0.15,
            (
                "Please look at it as soon as you can.",
                "At your earliest convenience, please.",
            ),
        ),
    ),
)

ANGRY = QuestionSpec(
    qid="angry",
    type="noul",
    instructions=("Is the customer angry?", "Does the writer sound upset?"),
    regions=(
        Region(
            "shouting",
            "stochastic",
            (0.1, 0.9),
            0.40,
            (
                "This is unacceptable!!",
                "I am really fed up!!",
                "Honestly, what is going on?!",
            ),
        ),
        Region("calm", "stochastic", (0.9, 0.1), 0.60),
    ),
)

REPEAT = QuestionSpec(
    qid="repeat_contact",
    type="noul",
    instructions=("Has the customer contacted us about this before?",),
    regions=(
        Region(
            "repeat",
            "deterministic",
            (0, 1),
            0.30,
            (
                "This is the third message I have sent about it.",
                "I already wrote to you last week about the same thing.",
            ),
        ),
        Region("first", "deterministic", (1, 0), 0.70),
    ),
)

QUESTIONS = (DEPARTMENT, URGENCY, ANGRY, REPEAT)


def render(rng, planted):
    sentences = [planted[q.qid] for q in QUESTIONS if q.qid in planted]
    sentences += rng.sample(FILLER, rng.randint(1, 3))
    rng.shuffle(sentences)
    return f"{rng.choice(GREETINGS)} " + " ".join(sentences)


DOMAIN = Domain("tickets", QUESTIONS, render, filler=GREETINGS + FILLER, hide=("urgency",))
