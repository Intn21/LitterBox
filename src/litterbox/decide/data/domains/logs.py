"""Log lines: semi-structured text with timestamps, numbers and tags.

The last line carries the event; lines before it are routine noise, and the
level word (INFO, WARN, ERROR) is drawn at random, so it predicts nothing. A
bare timeout is database half the time and network the other half. In an
unrecoverable corpus the severity clause is withheld, as if severity depended
on a service tier the log does not print.
"""

from __future__ import annotations

from litterbox.decide.data.regions import Domain, QuestionSpec, Region

LEVELS = ("INFO", "WARN", "ERROR")
ROUTINE = ("heartbeat ok", "config reloaded", "scheduled job finished", "cache warmed")

COMPONENT = QuestionSpec(
    qid="component",
    type="choice",
    instructions=("Which component is this event about?", "Assign this event to a component."),
    options=("database", "network", "storage", "auth"),
    regions=(
        Region(
            "database",
            "deterministic",
            (1, 0, 0, 0),
            0.20,
            (
                "query on table orders exceeded {n}ms",
                "replica lag is {n}s",
            ),
        ),
        Region(
            "network",
            "deterministic",
            (0, 1, 0, 0),
            0.20,
            (
                "packet loss {n}% on eth0",
                "upstream reset by peer after {n} retries",
            ),
        ),
        Region(
            "storage",
            "deterministic",
            (0, 0, 1, 0),
            0.20,
            (
                "disk usage at {n}% on /var",
                "write to volume failed: no space left",
            ),
        ),
        Region(
            "auth",
            "deterministic",
            (0, 0, 0, 1),
            0.20,
            (
                "token validation failed for user {n}",
                "too many failed logins from one address",
            ),
        ),
        Region(
            "bare_timeout",
            "stochastic",
            (0.5, 0.5, 0, 0),
            0.20,
            ("connection timed out after {n}ms",),
        ),
    ),
)

SEVERITY = QuestionSpec(
    qid="severity",
    type="score",
    instructions=("How severe is this event?", "Rate the severity."),
    options=("informational", "degraded", "outage"),
    regions=(
        Region(
            "outage",
            "deterministic",
            (0, 0, 1),
            0.25,
            (
                "all requests failing",
                "service unavailable",
            ),
        ),
        Region(
            "degraded",
            "deterministic",
            (0, 1, 0),
            0.30,
            (
                "latency elevated",
                "serving from fallback",
            ),
        ),
        Region("unstated", "deterministic", (1, 0, 0), 0.25),
        Region("partial", "stochastic", (0, 0.6, 0.4), 0.20, ("partial failures observed",)),
    ),
)

NEEDS_PAGE = QuestionSpec(
    qid="needs_page",
    type="noul",
    instructions=("Should the on-call engineer be paged?",),
    regions=(
        Region(
            "customer_facing",
            "stochastic",
            (0.15, 0.85),
            0.35,
            (
                "customer_facing=true",
                "tier=customer-facing",
            ),
        ),
        Region("internal", "stochastic", (0.95, 0.05), 0.65),
    ),
)

QUESTIONS = (COMPONENT, SEVERITY, NEEDS_PAGE)


def render(rng, planted):
    day, hour, minute = rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 50)
    service = f"svc-{rng.choice('abcdefgh')}{rng.randint(10, 99)}"

    def stamp(offset):
        second = rng.randint(0, 59)
        return f"2026-10-{day:02d}T{hour:02d}:{minute + offset:02d}:{second:02d}Z"

    lines = [
        f"{stamp(i)} {service} INFO {text}"
        for i, text in enumerate(rng.sample(ROUTINE, rng.randint(0, 2)))
    ]
    event = planted["component"].format(n=rng.randint(2, 900))
    if "severity" in planted:
        event += "; " + planted["severity"]
    if "needs_page" in planted:
        event += " " + planted["needs_page"]
    lines.append(f"{stamp(len(lines))} {service} {rng.choice(LEVELS)} {event}")
    return "\n".join(lines)


DOMAIN = Domain("logs", QUESTIONS, render, filler=LEVELS + ROUTINE, hide=("severity",))
