"""CLI Entry Point for VanillaRAG Evaluation Pipeline.

Supports:
- python -m evaluation.run_evaluation --limit 5
- python -m evaluation.run_evaluation --all
- python -m evaluation.run_evaluation --resume
- python -m evaluation.run_evaluation --retry-failed
- python -m evaluation.run_evaluation --metrics-only
- python -m evaluation.run_evaluation --query-ids Q001 Q002
- python -m evaluation.run_evaluation --category "Basic Retrieval"
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List

from tqdm import tqdm

from evaluation.config.evaluation_config import EvaluationConfig
from evaluation.dataset.dataset_loader import EvaluationQueryItem, load_evaluation_dataset
from evaluation.pipeline.evaluation_pipeline import EvaluationPipeline
from evaluation.reliability.checkpoint_manager import CheckpointManager, QueryCheckpointRecord
from evaluation.reporting.category_analysis import export_category_and_strategy_csvs
from evaluation.reporting.report_generator import generate_markdown_report, generate_summary_json
from evaluation.reporting.visualization import generate_evaluation_charts
from evaluation.utils.logging_utils import get_eval_logger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation of VanillaRAG retrieval and generation pipeline."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all 200 queries in the evaluation dataset.",
    )
    group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run a limited number of queries for smoke testing (e.g. --limit 5).",
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted evaluation run from the last completed query.",
    )
    group.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only previously failed or rate-limited queries.",
    )
    group.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute metrics, CSVs, reports, and charts from existing checkpoint data without running retrieval/LLM.",
    )

    parser.add_argument(
        "--query-ids",
        nargs="+",
        default=None,
        help="Run evaluation on specific query IDs (e.g. --query-ids Q001 Q006 Q046).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter evaluation to a specific query category.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="vanillarag_evaluation_dataset.json",
        help="Path to evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="evaluation/results",
        help="Root directory for saving checkpoints, results, metrics, and reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    logger = get_eval_logger("vanillarag_eval_cli")

    eval_config = EvaluationConfig(
        dataset_path=Path(args.dataset_path),
        results_dir=Path(args.results_dir),
    )
    eval_config.ensure_directories()

    checkpoint_mgr = CheckpointManager(eval_config.checkpoint_file)

    # 1. Handle --metrics-only mode
    if args.metrics_only:
        records = [r.to_dict() for r in checkpoint_mgr.get_all_records()]
        if not records:
            logger.error("No checkpoint records found in %s to compute metrics.", eval_config.checkpoint_file)
            sys.exit(1)

        logger.info("Computing metrics and reports from %d saved checkpoint records...", len(records))
        _generate_final_outputs(records, eval_config, logger)
        logger.info("Metrics-only processing completed successfully.")
        return

    # 2. Load dataset
    dataset = load_evaluation_dataset(eval_config.dataset_path)
    all_items: List[EvaluationQueryItem] = list(dataset)

    # 3. Determine target query items based on execution mode
    target_items: List[EvaluationQueryItem] = []

    if args.resume:
        completed_ids = checkpoint_mgr.get_completed_ids()
        target_items = [item for item in all_items if item.id not in completed_ids]
        logger.info(
            "Resume mode: Found %d completed queries out of %d. Running remaining %d queries.",
            len(completed_ids),
            len(all_items),
            len(target_items),
        )

    elif args.retry_failed:
        failed_ids = checkpoint_mgr.get_failed_ids()
        target_items = [item for item in all_items if item.id in failed_ids]
        logger.info(
            "Retry-failed mode: Found %d failed/rate-limited queries to retry.",
            len(target_items),
        )

    elif args.query_ids:
        target_ids = set(args.query_ids)
        target_items = [item for item in all_items if item.id in target_ids]
        logger.info("Selected %d queries by explicit IDs.", len(target_items))

    else:
        target_items = all_items
        if args.category:
            target_items = [item for item in target_items if item.category.lower() == args.category.lower()]
            logger.info("Filtered to %d queries in category '%s'.", len(target_items), args.category)

        if args.limit:
            target_items = target_items[: args.limit]
            logger.info("Limited evaluation execution to first %d queries.", len(target_items))
        elif not args.all:
            # Default behavior when no flag specified: run smoke test on 5 queries
            logger.info("No execution mode specified. Running quick smoke test on 5 queries (use --all to run all 200).")
            target_items = target_items[:5]

    if not target_items:
        logger.warning("No target queries to execute.")
        return

    # 4. Initialize Evaluation Pipeline
    pipeline = EvaluationPipeline(
        eval_config=eval_config,
        checkpoint_manager=checkpoint_mgr,
    )

    logger.info("Starting evaluation on %d queries...", len(target_items))
    per_query_records: List[QueryCheckpointRecord] = []

    # Open per_query JSONL for append
    with open(eval_config.per_query_jsonl, "a", encoding="utf-8") as jsonl_file:
        progress_bar = tqdm(target_items, desc="Evaluating Queries", unit="query")
        for item in progress_bar:
            progress_bar.set_postfix({"query_id": item.id, "cat": item.category[:15]})
            rec = pipeline.evaluate_query(item)
            per_query_records.append(rec)

            # Persist per-query record to JSONL
            jsonl_file.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            jsonl_file.flush()

    # 5. Generate aggregate reports and charts
    all_checkpoint_records = [r.to_dict() for r in checkpoint_mgr.get_all_records()]
    _generate_final_outputs(all_checkpoint_records, eval_config, logger)
    logger.info("Evaluation execution completed.")


def _generate_final_outputs(records: List[dict], eval_config: EvaluationConfig, logger) -> None:
    """Generate summary JSON, CSVs, markdown report, and charts."""
    summary_path = eval_config.metrics_dir / "summary_metrics.json"
    generate_summary_json(records, summary_path)
    logger.info("Generated summary metrics: %s", summary_path)

    csv_paths = export_category_and_strategy_csvs(records, eval_config.metrics_dir)
    logger.info("Generated CSV exports: %s", csv_paths)

    report_path = eval_config.reports_dir / "evaluation_report.md"
    generate_markdown_report(records, report_path)
    logger.info("Generated Markdown report: %s", report_path)

    chart_paths = generate_evaluation_charts(records, eval_config.charts_dir)
    logger.info("Generated visual charts: %s", list(chart_paths.keys()))


if __name__ == "__main__":
    main()

