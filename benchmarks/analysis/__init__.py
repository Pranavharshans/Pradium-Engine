"""Analysis: aggregation, statistics, reports, plots.

Strictly separated from measurement: everything here is derived from
``raw.jsonl`` and can be regenerated at any time, without rerunning inference.
"""

from .statistics import describe

__all__ = ["describe"]
