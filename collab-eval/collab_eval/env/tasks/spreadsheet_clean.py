"""
Spreadsheet cleanup task environment.

The agent receives a synthetic messy CSV with:
  - A merged-header artifact (row 2 is a units annotation, not data).
  - Inconsistent Revenue units ($K vs $M; two rows use $M notation).
  - Mixed date formats in the Notes column ("Jan 2023" vs "2023-04").
  - Two blank rows mid-table.

The task is to clean the CSV so it can be loaded into a data pipeline.

Tasks, documents, and failure cases in this artifact are fully synthetic and derived
from public evaluation patterns. No proprietary data or internal workflows are referenced.
"""

from __future__ import annotations

from pathlib import Path

from collab_eval.base import Episode, TaskEnv, TaskSpec
from collab_eval.graders import composite as composite_grader
from collab_eval.graders import deterministic as det


_DEFAULT_TASK_ID = "spreadsheet_clean_001"
_DEFAULT_INTENT = (
    "Clean this spreadsheet so it can be loaded into a data pipeline without errors."
)
_DEFAULT_CONSTRAINTS = [
    "Preserve all original data values (do not drop rows with real data).",
    "Output valid CSV.",
    "Normalize all Revenue and OpEx values to $K throughout.",
]

_WEIGHTS: dict[str, float] = {
    "data_preservation": 0.35,
    "format_validity": 0.25,
    "unit_consistency": 0.25,
    "completeness": 0.15,
}

# 12 real data rows (after removing blank rows and the merged-header annotation).
# Design note: we hardcode the expected count rather than computing it from the
# raw input, because the agent does not know which rows we consider "real".
_EXPECTED_DATA_ROWS = 12


def load_default_task(data_dir: Path | None = None) -> TaskSpec:
    """
    Load the default spreadsheet cleanup task from the sample data directory.

    Resolves data_dir relative to this file if not provided.
    """
    if data_dir is None:
        data_dir = Path(__file__).parents[3] / "data" / "sample_docs"
    input_doc = (data_dir / "financial_summary_messy.csv").read_text()
    return TaskSpec(
        task_id=_DEFAULT_TASK_ID,
        task_type="spreadsheet_clean",
        input_doc=input_doc,
        intent=_DEFAULT_INTENT,
        constraints=_DEFAULT_CONSTRAINTS,
        source_docs=[],
    )


class SpreadsheetCleanEnv(TaskEnv):
    """
    Task environment for spreadsheet cleanup.

    Grading is primarily deterministic: CSV validity, row preservation, and
    unit normalization are all algorithmically checkable. Completeness (no
    silent column dropping) is checked by comparing header sets.
    """

    def __init__(self, task: TaskSpec | None = None) -> None:
        super().__init__()
        self._task = task or load_default_task()

    def reset(self) -> TaskSpec:
        """Return the spreadsheet cleanup task spec."""
        self._current_spec = self._task
        return self._current_spec

    def grade(self, spec: TaskSpec, output: str) -> dict[str, float]:
        """
        Score output on four dimensions using deterministic checks.

        data_preservation:  Were the real data rows retained?
        format_validity:    Is the output parseable CSV?
        unit_consistency:   Are all monetary values in $K (no $M notation)?
        completeness:       Are all expected columns present?
        """
        # format_validity: must be parseable CSV.
        parseable = det.csv_parseable(output)
        format_score = 1.0 if parseable else 0.0

        if not parseable:
            # Cannot score other dimensions if CSV is malformed.
            return {
                "data_preservation": 0.0,
                "format_validity": 0.0,
                "unit_consistency": 0.0,
                "completeness": 0.0,
            }

        # data_preservation: check row count against expected.
        # Design note: row_count_preserved checks whether the number of non-header,
        # non-blank rows matches the expected count. This is gameable by duplicating
        # rows, but catches the most common failure mode: silent row dropping.
        preservation_score = (
            1.0
            if det.row_count_preserved(output, expected_data_rows=_EXPECTED_DATA_ROWS)
            else 0.0
        )

        # unit_consistency: no "$M" or "M" as a unit suffix should remain.
        unit_score = 1.0 if det.unit_normalized(output, forbidden_pattern=r"\$M|\bM\b") else 0.0

        # completeness: all original columns present.
        original_headers = {"Quarter", "Revenue", "OpEx", "Headcount", "Notes"}
        completeness_score = det.headers_preserved(output, required_headers=original_headers)

        return {
            "data_preservation": preservation_score,
            "format_validity": format_score,
            "unit_consistency": unit_score,
            "completeness": round(completeness_score, 3),
        }

    def step(self, agent_output: str) -> Episode:
        """
        Grade agent output and apply hard-fail for non-parseable CSV.

        Flag: csv_not_parseable — output is not valid CSV at all.
        """
        if self._current_spec is None:
            raise RuntimeError("Call reset() before step().")
        spec = self._current_spec
        scores = self.grade(spec, agent_output)
        flags: list[str] = []

        if not det.csv_parseable(agent_output):
            flags.append("csv_not_parseable")

        composite = composite_grader.weighted_score(scores, _WEIGHTS)
        if flags:
            composite = composite_grader.apply_hard_fail_cap(composite, cap=0.2)

        return Episode(
            spec=spec,
            agent_output=agent_output,
            grader_scores=scores,
            composite_score=round(composite, 3),
            flags=flags,
        )
