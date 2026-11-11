"""Category-wise and Strategy-wise metrics export to CSV."""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from evaluation.metrics.aggregate_metrics import (
    aggregate_category_metrics,
    aggregate_strategy_metrics,
)


def export_category_and_strategy_csvs(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Export category_metrics.csv and strategy_analysis.csv."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cat_data = aggregate_category_metrics(records)
    cat_df = pd.DataFrame(cat_data)
    cat_csv_path = output_dir / "category_metrics.csv"
    cat_df.to_csv(cat_csv_path, index=False, encoding="utf-8")

    strat_data = aggregate_strategy_metrics(records)
    strat_df = pd.DataFrame(strat_data)
    strat_csv_path = output_dir / "strategy_analysis.csv"
    strat_df.to_csv(strat_csv_path, index=False, encoding="utf-8")

    return {
        "category_metrics_csv": cat_csv_path,
        "strategy_analysis_csv": strat_csv_path,
    }

