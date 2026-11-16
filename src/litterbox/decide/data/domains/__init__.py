"""The generated domains: three shapes of state a decision model has to read."""

from litterbox.decide.data.domains import logs, orders, tickets

DOMAINS = {d.name: d for d in (tickets.DOMAIN, logs.DOMAIN, orders.DOMAIN)}

__all__ = ["DOMAINS"]
