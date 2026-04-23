"""
Deterministic grader functions.

Each function is a rule-based check that can run without any model calls.
Where a heuristic is known to be imperfect or gameable, a '# Design note:' comment
explains the failure mode and why we use it anyway.

Tasks, documents, and failure cases in this artifact are fully synthetic and derived
from public evaluation patterns. No proprietary data or internal workflows are referenced.
"""

from __future__ import annotations

import io
import re


# ── Word count ────────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    """Return the number of whitespace-delimited tokens in text."""
    return len(text.split())


def word_count_check(output: str, max_words: int) -> float:
    """
    Return 1.0 if output is within max_words, else decay toward 0.0.

    Score is linear between max_words and max_words*1.5, then 0.0 beyond.

    Design note: a hard binary check (pass/fail at max_words) creates a cliff
    that encourages agents to land at exactly max_words - 1. A soft decay
    is harder to game precisely but still penalizes large violations.
    Gameable by: compressing whitespace or using contractions to hit the limit
    while preserving equivalent information — this is generally acceptable behavior.
    """
    n = word_count(output)
    if n <= max_words:
        return 1.0
    overflow = n - max_words
    cliff = max_words * 0.5
    if overflow >= cliff:
        return 0.0
    return round(1.0 - (overflow / cliff), 3)


# ── Passive voice ─────────────────────────────────────────────────────────────

# Matches common auxiliary + past-participle constructions.
# Design note: this regex catches the most common passive patterns but misses
# complex forms like "is being revised" or passives with modal auxiliaries
# ("should be reviewed"). It also produces false positives on adjective uses
# ("is concerned about"). Good enough as a cheap proxy; not a reliable parser.
# Gameable by: rewording passives into unusual active constructions or using
# modals that the regex does not capture.
_PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+\w+ed\b",
    re.IGNORECASE,
)


def passive_voice_ratio(text: str) -> float:
    """
    Return the fraction of sentences that contain a passive construction.

    Returns 0.0 if the text has no sentences.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    passive_count = sum(1 for s in sentences if _PASSIVE_RE.search(s))
    return round(passive_count / len(sentences), 3)


# ── CSV checks ────────────────────────────────────────────────────────────────

def csv_parseable(output: str) -> bool:
    """
    Return True if output can be parsed as CSV by the standard library.

    Design note: pandas would be more robust here, but the standard library
    is sufficient for the failure mode we care about: completely malformed output.
    Gameable by: producing syntactically valid CSV that doesn't match the
    expected schema — unit_normalized and headers_preserved catch that.
    """
    import csv
    try:
        reader = csv.reader(io.StringIO(output.strip()))
        rows = list(reader)
        return len(rows) >= 2  # at least header + one data row
    except Exception:
        return False


def row_count_preserved(output: str, expected_data_rows: int) -> bool:
    """
    Return True if the output CSV has at least expected_data_rows non-header rows.

    Blank rows are excluded from the count.

    Design note: checks that the agent did not silently drop rows. Does not
    verify that the rows contain the correct values — that would require a
    per-cell comparison which is expensive and brittle for messy input data.
    Gameable by: duplicating rows to inflate count while dropping originals.
    """
    import csv
    try:
        reader = csv.reader(io.StringIO(output.strip()))
        rows = [r for r in reader if any(cell.strip() for cell in r)]
        # Subtract one for the header row.
        data_rows = len(rows) - 1
        return data_rows >= expected_data_rows
    except Exception:
        return False


def unit_normalized(output: str, forbidden_pattern: str) -> bool:
    """
    Return True if the output contains no matches for forbidden_pattern.

    Used to detect lingering $M notation when the task requires $K throughout.

    Design note: regex-based unit checking is gameable by hiding the unit
    in free-text Notes fields. A cell-by-cell numeric parser would be more
    reliable but significantly more complex. The heuristic catches the most
    common failure mode: an agent that simply omits the conversion.
    """
    return not bool(re.search(forbidden_pattern, output))


def headers_preserved(output: str, required_headers: set[str]) -> float:
    """
    Return the fraction of required_headers present in the output CSV's header row.

    Case-insensitive comparison.

    Design note: checks completeness of the output schema. Does not verify
    column order, which some downstream pipelines care about — intentionally
    out of scope for this heuristic.
    """
    import csv
    try:
        reader = csv.reader(io.StringIO(output.strip()))
        header_row = next(reader, [])
        output_headers = {h.strip().lower() for h in header_row}
        required_lower = {h.lower() for h in required_headers}
        if not required_lower:
            return 1.0
        found = len(output_headers & required_lower)
        return round(found / len(required_lower), 3)
    except Exception:
        return 0.0


# ── Citation checks ───────────────────────────────────────────────────────────

# Matches common inline citation patterns: (Source A), [1], [A], (1), etc.
# Design note: this regex is deliberately broad. It catches the most common
# conventions without imposing a specific citation format. Gameable by
# inserting empty brackets or parentheses — citation_accurate (LLM) catches that.
_CITATION_RE = re.compile(
    r"(\[[^\]]{1,30}\]|\([Ss]ource\s+[A-Za-z0-9]\)|\(\d+\))",
)


def citation_present(output: str, required_count: int) -> float:
    """
    Return a score in [0.0, 1.0] based on how many citation markers are found.

    Score is required_found / required_count, capped at 1.0.

    Design note: counts markers, not unique sources. An agent that repeats
    one citation twice to satisfy a two-citation requirement will score 1.0
    here. The citation_accurate dimension (LLM-based) catches source diversity
    failures. This division of labor keeps each grader cheap and auditable.
    """
    matches = _CITATION_RE.findall(output)
    found = len(matches)
    if required_count == 0:
        return 1.0
    return round(min(found / required_count, 1.0), 3)
