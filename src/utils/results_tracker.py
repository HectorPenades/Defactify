"""Accumulates experiment results into a comparison CSV and Markdown table."""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


_COLUMNS = [
    'date', 'experiment', 'task', 'mode', 'architecture',
    'accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall',
    'duration_s', 'checkpoint',
]

_TABLE_PATH = Path('results/comparison_table.csv')
_MD_PATH = Path('results/comparison_table.md')


def _load_rows() -> List[Dict[str, str]]:
    """Load existing rows from CSV."""
    if not _TABLE_PATH.exists():
        return []
    with open(_TABLE_PATH, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def _save_rows(rows: List[Dict[str, str]]) -> None:
    """Write all rows to CSV."""
    _TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TABLE_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: List[Dict[str, str]]) -> None:
    """Write markdown comparison table."""
    lines = []
    lines.append("# Experiment Comparison Table")
    lines.append(f"\n_Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

    header = "| Date | Experiment | Task | Mode | Arch | Accuracy | F1-Macro | F1-Weighted | Precision | Recall | Duration(s) |"
    sep    = "|------|------------|------|------|------|----------|----------|-------------|-----------|--------|-------------|"
    lines.append(header)
    lines.append(sep)

    for r in rows:
        def fmt(key, decimals=4):
            v = r.get(key, '')
            try:
                return f"{float(v):.{decimals}f}"
            except (ValueError, TypeError):
                return v

        lines.append(
            f"| {r.get('date','')} "
            f"| {r.get('experiment','')} "
            f"| {r.get('task','')} "
            f"| {r.get('mode','')} "
            f"| {r.get('architecture','')} "
            f"| {fmt('accuracy')} "
            f"| {fmt('f1_macro')} "
            f"| {fmt('f1_weighted')} "
            f"| {fmt('precision')} "
            f"| {fmt('recall')} "
            f"| {fmt('duration_s', 0)} |"
        )

    _MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_MD_PATH, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def record_result(config: Dict[str, Any], test_metrics: Dict[str, Any],
                  checkpoint_path: str, duration_s: float) -> None:
    """
    Append (or update) one experiment row in the comparison table.

    If an experiment with the same name already exists, its row is replaced
    so repeated runs always reflect the latest result.

    Args:
        config: Experiment configuration dict
        test_metrics: Dict with accuracy, f1_macro, f1_weighted, precision, recall
        checkpoint_path: Path to best checkpoint
        duration_s: Total experiment duration in seconds
    """
    exp_name = config.get('experiment_name', 'unknown')
    row = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'experiment': exp_name,
        'task': config.get('task', ''),
        'mode': config.get('dataset', {}).get('mode', ''),
        'architecture': config.get('model', {}).get('architecture', ''),
        'accuracy': round(float(test_metrics.get('accuracy', 0)), 6),
        'f1_macro': round(float(test_metrics.get('f1_macro', 0)), 6),
        'f1_weighted': round(float(test_metrics.get('f1_weighted', 0)), 6),
        'precision': round(float(test_metrics.get('precision', 0)), 6),
        'recall': round(float(test_metrics.get('recall', 0)), 6),
        'duration_s': round(duration_s, 1),
        'checkpoint': checkpoint_path,
    }

    rows = _load_rows()
    # Replace row if experiment already exists
    rows = [r for r in rows if r.get('experiment') != exp_name]
    rows.append(row)
    # Sort by experiment name for stable ordering
    rows.sort(key=lambda r: r.get('experiment', ''))

    _save_rows(rows)
    _write_markdown(rows)
    print(f"[ResultsTracker] Table updated: {_TABLE_PATH}")
    print(f"[ResultsTracker] Markdown:      {_MD_PATH}")
