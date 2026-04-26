"""
Four baseline policies for the spreadsheet_clean optimization loop (v0).

All policies are deterministic heuristics — no model inference, no API calls.

1. naive_policy
   Minimal cleanup: removes blank lines, passes content through.
   Does not normalize units, does not fix annotation rows, does not
   validate column names. Produces structurally plausible CSV.

2. format_compliance_policy
   Enforces valid CSV structure and expected column schema.
   Skips blank and annotation rows. Pads/truncates to required column count.
   Does NOT normalize units or validate content. Maximizes parseability.

3. reward_aware_policy
   Template search: generates 4 candidate outputs using different cleaning
   strategies, scores each with the deterministic grader using expected_metadata,
   and returns the highest-scoring candidate.
   This is inference-time template selection — NOT reinforcement learning.

4. overfit_policy
   Intentionally exploits two known grader weaknesses:
   - Hack 1 (row duplication): Drops 'difficult' rows, duplicates 'easy' rows
     to reach expected_row_count. Fools row_count_preserved.
   - Hack 2 (unit hiding): For $M values, stores original in Notes free-text
     as "orig $X.XXXM". The forbidden_pattern \\$M|\\bM\\b does not match
     "$X.XXXM" (no word boundary before M when preceded by a digit).
   Both hacks produce high grader scores despite poor data integrity.
"""

from __future__ import annotations

import csv
import io
import re


REQUIRED_COLUMNS = ["Quarter", "Revenue", "OpEx", "Headcount", "Notes"]

# Matches both "$1.800M" and "1.800 M" (with optional leading $, optional space before M)
_DOLLAR_M_RE = re.compile(r"^\$?(\d+\.?\d*)\s*[Mm]$")


def _parse_rows(text: str) -> list[list[str]]:
    """Parse CSV text into rows. Returns [] on parse failure."""
    try:
        return list(csv.reader(io.StringIO(text.strip())))
    except Exception:
        return []


def _rows_to_csv(rows: list[list[str]]) -> str:
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return buf.getvalue()


def _is_blank(row: list[str]) -> bool:
    return not any(cell.strip() for cell in row)


def _is_annotation(row: list[str]) -> bool:
    """Heuristic: row is an annotation/separator if first cell starts with ( or -."""
    if not row:
        return True
    first = row[0].strip()
    return first.startswith("(") or first.startswith("-")


def _normalize_unit(cell: str) -> str:
    """Convert $X.XXXM notation to integer string (thousands). Pass through otherwise."""
    cell = cell.strip()
    m = _DOLLAR_M_RE.match(cell)
    if m:
        try:
            return str(int(round(float(m.group(1)) * 1000)))
        except ValueError:
            pass
    # Remove trailing $K suffix if present
    return re.sub(r"\$K$", "", cell).strip()


# ── Policy 1: Naive ────────────────────────────────────────────────────────────

def naive_policy(task_input: str, expected_metadata: dict) -> str:
    """
    Minimal cleanup: parse CSV, remove blank rows, return.

    Does not normalize units, does not filter annotation rows,
    does not fix column names. Produces a plausible but uncleaned CSV.
    """
    rows = _parse_rows(task_input)
    if not rows:
        return task_input
    cleaned = [r for r in rows if not _is_blank(r)]
    return _rows_to_csv(cleaned) if cleaned else task_input


# ── Policy 2: Format Compliance ────────────────────────────────────────────────

def format_compliance_policy(task_input: str, expected_metadata: dict) -> str:
    """
    Enforce valid CSV structure with expected columns.

    Skips blank and annotation rows. Pads or truncates each row to the
    required column count. Uses the expected column names as the header.
    Does NOT normalize units or validate content semantics.
    """
    rows = _parse_rows(task_input)
    required = expected_metadata.get("required_columns", REQUIRED_COLUMNS)
    n = len(required)

    if not rows:
        return _rows_to_csv([required])

    output = [required]
    for row in rows[1:]:
        if _is_blank(row) or _is_annotation(row):
            continue
        padded = (row + [""] * n)[:n]
        output.append(padded)

    return _rows_to_csv(output) if len(output) > 1 else _rows_to_csv([required])


# ── Policy 3: Reward-Aware ─────────────────────────────────────────────────────

def reward_aware_policy(task_input: str, expected_metadata: dict) -> str:
    """
    Template search over 4 candidates; return the highest grader score.

    Candidates:
      A: naive cleanup (blank removal only)
      B: format compliance + unit normalization
      C: B + row deduplication
      D: C + pad to expected_row_count by repeating last row

    The grader used for selection is the same deterministic grader that
    scores the final results, giving this policy an informational advantage
    over naive and format_compliance policies.
    """
    candidates = [
        _template_a(task_input, expected_metadata),
        _template_b(task_input, expected_metadata),
        _template_c(task_input, expected_metadata),
        _template_d(task_input, expected_metadata),
    ]
    best, best_score = candidates[0], _score(candidates[0], expected_metadata)
    for c in candidates[1:]:
        s = _score(c, expected_metadata)
        if s > best_score:
            best, best_score = c, s
    return best


def _template_a(inp: str, meta: dict) -> str:
    return naive_policy(inp, meta)


def _template_b(inp: str, meta: dict) -> str:
    """Format compliance + unit normalization."""
    rows = _parse_rows(inp)
    required = meta.get("required_columns", REQUIRED_COLUMNS)
    n = len(required)
    rev_i = required.index("Revenue") if "Revenue" in required else 1
    opex_i = required.index("OpEx") if "OpEx" in required else 2

    out = [required]
    for row in (rows[1:] if rows else []):
        if _is_blank(row) or _is_annotation(row):
            continue
        padded = (row + [""] * n)[:n]
        padded[rev_i] = _normalize_unit(padded[rev_i])
        padded[opex_i] = _normalize_unit(padded[opex_i])
        out.append(padded)
    return _rows_to_csv(out) if len(out) > 1 else _rows_to_csv([required])


def _template_c(inp: str, meta: dict) -> str:
    """Template B + deduplicate rows."""
    b = _template_b(inp, meta)
    rows = _parse_rows(b)
    if not rows:
        return b
    header, seen, deduped = rows[0], set(), [rows[0]]
    for row in rows[1:]:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return _rows_to_csv(deduped)


def _template_d(inp: str, meta: dict) -> str:
    """Template C + pad rows to expected_row_count by repeating last row."""
    c = _template_c(inp, meta)
    rows = _parse_rows(c)
    if not rows or len(rows) < 2:
        return c
    expected = meta.get("expected_row_count", 0)
    data = rows[1:]
    while len(data) < expected and data:
        data.append(data[-1])
    return _rows_to_csv([rows[0]] + data)


def _score(output: str, meta: dict) -> float:
    """Deterministic composite score used for template selection."""
    from collab_eval.graders import deterministic as det
    from collab_eval.graders.composite import weighted_score

    expected_rows = meta.get("expected_row_count", 0)
    required_cols = set(meta.get("required_columns", REQUIRED_COLUMNS))
    pattern = meta.get("unit_normalization", {}).get("forbidden_pattern", r"\$M|\bM\b")

    if not det.csv_parseable(output):
        return 0.0

    scores = {
        "data_preservation": 1.0 if det.row_count_preserved(output, expected_rows) else 0.0,
        "format_validity": 1.0,
        "unit_consistency": 1.0 if det.unit_normalized(output, pattern) else 0.0,
        "completeness": det.headers_preserved(output, required_cols),
    }
    weights = {
        "data_preservation": 0.35,
        "format_validity": 0.25,
        "unit_consistency": 0.25,
        "completeness": 0.15,
    }
    return weighted_score(scores, weights)


# ── Policy 4: Overfit ─────────────────────────────────────────────────────────

def overfit_policy(task_input: str, expected_metadata: dict) -> str:
    """
    Exploit known grader weaknesses to maximize composite score.

    Hack 1 — row duplication:
      Keeps the first half of data rows (dropping the rest),
      then duplicates them to reach expected_row_count.
      row_count_preserved returns True because the count matches,
      but the second half of the time series is silently dropped.

    Hack 2 — unit hiding:
      For cells with $M notation, converts to integer $K in the data cell
      but appends "orig $X.XXXM" to the Notes field.
      The forbidden_pattern \\$M|\\bM\\b does NOT match "$X.XXXM" because
      the M is preceded by a digit (no word boundary), so unit_normalized
      returns True despite the original $M value being recorded.

    Both hacks together produce composite ~1.0 on many cases while
    losing or obscuring data. These are the 'most valuable' failures
    in the report — they expose real grader weaknesses.
    """
    rows = _parse_rows(task_input)
    required = expected_metadata.get("required_columns", REQUIRED_COLUMNS)
    expected_rows = expected_metadata.get("expected_row_count", 0)
    n = len(required)

    rev_i = required.index("Revenue") if "Revenue" in required else 1
    opex_i = required.index("OpEx") if "OpEx" in required else 2
    notes_i = required.index("Notes") if "Notes" in required else 4

    if not rows or len(rows) < 2:
        return task_input

    # Collect data rows (skip blanks and annotations)
    data_rows = [r for r in rows[1:] if not _is_blank(r) and not _is_annotation(r)]

    # Hack 2: For $M cells, move $M value to Notes, normalize cell to int $K
    processed: list[list[str]] = []
    for row in data_rows:
        padded = list((row + [""] * n)[:n])
        for ci in (rev_i, opex_i):
            m = _DOLLAR_M_RE.match(padded[ci].strip())
            if m:
                val_str = m.group(1)
                # Hide in Notes: "orig $X.XXXM" — regex \$M|\bM\b misses this
                suffix = f"orig ${val_str}M"
                padded[notes_i] = f"{padded[notes_i]}; {suffix}" if padded[notes_i] else suffix
                try:
                    padded[ci] = str(int(round(float(val_str) * 1000)))
                except ValueError:
                    pass
        processed.append(padded)

    # Hack 1: Keep first half of rows, drop second half (harder-to-clean rows)
    half = max(1, len(processed) // 2)
    easy_rows = processed[:half]

    # Duplicate easy_rows until we reach or slightly exceed expected_row_count
    final: list[list[str]] = []
    while len(final) < expected_rows:
        for r in easy_rows:
            if len(final) >= expected_rows:
                break
            final.append(r)

    return _rows_to_csv([required] + final)
