"""Sparse embedding generation for existing Qdrant chunks with checkpointing and resume support.

Generates BM25 sparse vectors (indices and values) using FastEmbed ('Qdrant/bm25')
for all existing points in the persistent collection (e.g. ~123,776 points).

Supports:
- Streaming scroll through Qdrant without loading dense vectors into memory (with_vectors=False).
- Incremental JSONL checkpointing to disk for offline execution or Google Colab export.
- Seamless resume capability via state tracking so interrupted runs never recompute.
- Configurable batch size, progress reporting, elapsed time, throughput, and ETA.
- Completely non-destructive by default: only saves checkpoints to disk unless --apply-to-qdrant is explicitly given.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.config import PipelineConfig
from src.ingestion.logging_config import logger
from src.ingestion.sparse_embeddings import LocalBM25SparseEmbeddings, SparseVectorResult
from src.ingestion.vector_store import QdrantManager


class CheckpointManager:
    """Manages disk checkpoints and resume state for generated sparse vectors."""

    def __init__(self, checkpoint_dir: Path, collection_name: str, sparse_name: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.sparse_name = sparse_name
        self.state_file = self.checkpoint_dir / f"checkpoint_state_{collection_name}.json"
        self.processed_ids: Set[str] = set()
        self.current_part_index = 1
        self._load_state()

    def _load_state(self):
        """Load state and collect all processed point IDs from existing checkpoints."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.current_part_index = state.get("current_part_index", 1)
                logger.info(
                    f"Found existing checkpoint state for collection '{self.collection_name}'. "
                    f"Saved part index: {self.current_part_index}"
                )
            except Exception as e:
                logger.warning(f"Could not read checkpoint state file: {e}")

        # Scan all existing jsonl checkpoint files to populate processed_ids set
        checkpoint_files = sorted(self.checkpoint_dir.glob("sparse_part_*.jsonl"))
        total_loaded = 0
        for cp_file in checkpoint_files:
            try:
                with open(cp_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            if "id" in record:
                                self.processed_ids.add(str(record["id"]))
                                total_loaded += 1
            except Exception as e:
                logger.warning(f"Error reading checkpoint file {cp_file.name}: {e}")

        if total_loaded > 0:
            logger.info(
                f"Resume support: loaded {len(self.processed_ids):,} existing processed point IDs "
                f"from {len(checkpoint_files)} checkpoint file(s)."
            )

    def is_processed(self, point_id: str) -> bool:
        return str(point_id) in self.processed_ids

    def save_batch(
        self,
        batch_records: List[Dict[str, Any]],
        last_offset: Optional[Any] = None,
        total_in_collection: int = 0,
    ):
        """Append a batch of sparse vector records to the current checkpoint file."""
        if not batch_records:
            return

        part_file = self.checkpoint_dir / f"sparse_part_{self.current_part_index:05d}.jsonl"

        # Rotate file if it exceeds ~25,000 records to keep files manageable
        if part_file.exists() and part_file.stat().st_size > 25 * 1024 * 1024:
            self.current_part_index += 1
            part_file = self.checkpoint_dir / f"sparse_part_{self.current_part_index:05d}.jsonl"

        with open(part_file, "a", encoding="utf-8") as f:
            for record in batch_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.processed_ids.add(str(record["id"]))

        # Atomically write state metadata
        state_data = {
            "collection_name": self.collection_name,
            "sparse_vector_name": self.sparse_name,
            "total_points_in_collection": total_in_collection,
            "total_processed": len(self.processed_ids),
            "current_part_index": self.current_part_index,
            "last_offset": str(last_offset) if last_offset is not None else None,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        temp_state = self.state_file.with_suffix(".tmp")
        with open(temp_state, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)
            f.flush()
        temp_state.replace(self.state_file)


def extract_point_text(payload: Optional[Dict[str, Any]]) -> str:
    """Extract clean text content from point payload with fallbacks."""
    if not payload:
        return ""
    text = payload.get("page_content") or payload.get("text") or ""
    if not text:
        # Fallback to description or error message if available
        text = payload.get("description") or payload.get("failure_reason") or ""
    return str(text)


def run_sparse_generation(
    collection_name: Optional[str] = None,
    input_file: Optional[str] = None,
    model_name: Optional[str] = None,
    sparse_name: Optional[str] = None,
    batch_size: int = 128,
    checkpoint_dir: Optional[str] = None,
    save_checkpoints: bool = True,
    apply_to_qdrant: bool = False,
    resume: bool = True,
    sample_limit: Optional[int] = None,
    threads: Optional[int] = None,
):
    """Main routine to generate sparse BM25 embeddings for chunks with checkpointing."""
    config = PipelineConfig()
    if collection_name:
        config.qdrant_collection = collection_name
    if model_name:
        config.sparse_model_name = model_name
    if sparse_name:
        config.sparse_vector_name = sparse_name
    if checkpoint_dir:
        config.sparse_checkpoint_dir = Path(checkpoint_dir)

    manager = None
    client = None

    print("\n" + "=" * 80)
    print("SPARSE EMBEDDING GENERATION PIPELINE (BM25)")
    print("=" * 80)
    print(f"Target Collection:       {config.qdrant_collection}")
    print(f"Input Source:            {input_file if input_file else 'Direct Qdrant Scroll'}")
    print(f"Sparse Model:            {config.sparse_model_name}")
    print(f"Sparse Vector Name:      {config.sparse_vector_name}")
    print(f"Batch Size:              {batch_size}")
    print(f"Checkpoint Directory:    {config.sparse_checkpoint_dir}")
    print(f"Save Checkpoints to Disk:{save_checkpoints}")
    print(f"Apply Directly to Qdrant:{apply_to_qdrant} (Non-destructive: False by default)")
    print(f"Resume Enabled:          {resume}")
    if sample_limit:
        print(f"Sample Limit:            {sample_limit:,}")
    print("=" * 80)

    # 1. Connect to Qdrant if needed
    total_points = 0
    if not input_file or apply_to_qdrant:
        try:
            manager = QdrantManager(config)
            client = manager.client
            info = manager.get_collection_info()
            total_points = info.points_count or 0
            print(f"Total points stored in collection: {total_points:,}")
        except Exception as e:
            if not input_file:
                logger.error(f"Cannot access collection '{config.qdrant_collection}': {e}")
                print(f"❌ Error: Cannot access collection '{config.qdrant_collection}': {e}")
                return
            else:
                print(f"Notice: Running in offline mode without active Qdrant connection ({e}).")

    target_total = min(total_points, sample_limit) if (total_points > 0 and sample_limit) else total_points

    # 2. Check if applying directly to Qdrant requires schema migration
    if apply_to_qdrant:
        existing_sparse = info.config.params.sparse_vectors or {}
        if config.sparse_vector_name not in existing_sparse:
            print(f"Notice: Registering sparse vector '{config.sparse_vector_name}' in collection schema...")
            manager.ensure_sparse_vector(config.sparse_vector_name)

    # 3. Setup Checkpoint Manager
    cp_mgr = CheckpointManager(
        checkpoint_dir=config.sparse_checkpoint_dir,
        collection_name=manager.collection_name if manager else config.qdrant_collection,
        sparse_name=config.sparse_vector_name,
    )

    already_done = len(cp_mgr.processed_ids) if resume else 0
    if already_done > 0:
        print(f"ℹ️ Resume state: {already_done:,} / {target_total:,} points already processed in checkpoints.")
        if already_done >= target_total:
            print("✅ All target points have already been processed into checkpoints!")
            print(f"Checkpoints directory: {config.sparse_checkpoint_dir.resolve()}")
            return

    # 4. Initialize Sparse Embedder
    print(f"\nInitializing LocalBM25SparseEmbeddings ({config.sparse_model_name})...")
    embedder = LocalBM25SparseEmbeddings(
        config=config,
        model_name=config.sparse_model_name,
        batch_size=batch_size,
        threads=threads,
    )

    # 5. Streaming Loop (either from input file or Qdrant scroll)
    print("\nStarting sparse vector generation...")
    print("-" * 80)

    overall_start = time.perf_counter()
    processed_in_session = 0
    next_offset = None
    batch_num = 0

    def stream_items():
        """Yield (point_id, text) tuples from input_file or Qdrant scroll."""
        nonlocal next_offset
        if input_file:
            in_path = Path(input_file)
            if not in_path.exists():
                raise FileNotFoundError(f"Input file '{in_path}' does not exist.")
            with open(in_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pt_id = str(rec.get("id") or rec.get("chunk_id"))
                    text = str(rec.get("text") or rec.get("page_content") or "")
                    yield pt_id, text
        else:
            while True:
                fetch_limit = min(batch_size, 500)
                pts, next_offset = client.scroll(
                    collection_name=manager.collection_name,
                    limit=fetch_limit,
                    offset=next_offset,
                    with_payload=["page_content", "text", "description", "failure_reason"],
                    with_vectors=False,
                )
                if not pts:
                    break
                for pt in pts:
                    yield str(pt.id), extract_point_text(pt.payload)
                if next_offset is None:
                    break

    try:
        buffer_ids: List[str] = []
        buffer_texts: List[str] = []

        for pt_id, text in stream_items():
            if resume and cp_mgr.is_processed(pt_id):
                continue

            buffer_ids.append(pt_id)
            buffer_texts.append(text)

            if len(buffer_ids) >= batch_size:
                batch_num += 1
                batch_start = time.perf_counter()

                sparse_results = embedder.embed_texts(buffer_texts)

                checkpoint_records = [
                    {"id": pid, "indices": s.indices, "values": s.values}
                    for pid, s in zip(buffer_ids, sparse_results)
                ]

                if save_checkpoints:
                    cp_mgr.save_batch(
                        checkpoint_records,
                        last_offset=next_offset,
                        total_in_collection=target_total,
                    )

                if apply_to_qdrant and manager:
                    q_updates = [
                        (pid, s.indices, s.values)
                        for pid, s in zip(buffer_ids, sparse_results)
                    ]
                    manager.update_point_sparse_vectors(
                        updates=q_updates,
                        vector_name=config.sparse_vector_name,
                    )

                processed_in_session += len(buffer_ids)
                total_completed = already_done + processed_in_session

                batch_time = time.perf_counter() - batch_start
                elapsed = time.perf_counter() - overall_start
                rate = processed_in_session / elapsed if elapsed > 0 else 0.0

                if target_total > 0:
                    remaining_pts = max(0, target_total - total_completed)
                    eta_seconds = remaining_pts / rate if rate > 0 else 0.0
                    pct_str = f"({(total_completed / target_total) * 100:.1f}%)"
                    eta_str = f"{eta_seconds / 60:.1f}m"
                else:
                    pct_str = ""
                    eta_str = "calculating..."

                print(
                    f"[Batch {batch_num:,}] Processed: {total_completed:,} {pct_str} | "
                    f"Rate: {rate:.1f} pts/s | "
                    f"Batch: {batch_time:.2f}s | "
                    f"Elapsed: {elapsed / 60:.1f}m | "
                    f"ETA: {eta_str}"
                )

                buffer_ids.clear()
                buffer_texts.clear()

                if sample_limit and total_completed >= sample_limit:
                    break

                if batch_num % 10 == 0:
                    gc.collect()

        # Flush any remaining items in buffer
        if buffer_ids and (not sample_limit or (already_done + processed_in_session) < sample_limit):
            batch_num += 1
            sparse_results = embedder.embed_texts(buffer_texts)
            checkpoint_records = [
                {"id": pid, "indices": s.indices, "values": s.values}
                for pid, s in zip(buffer_ids, sparse_results)
            ]
            if save_checkpoints:
                cp_mgr.save_batch(
                    checkpoint_records,
                    last_offset=next_offset,
                    total_in_collection=target_total,
                )
            if apply_to_qdrant and manager:
                q_updates = [
                    (pid, s.indices, s.values)
                    for pid, s in zip(buffer_ids, sparse_results)
                ]
                manager.update_point_sparse_vectors(
                    updates=q_updates,
                    vector_name=config.sparse_vector_name,
                )
            processed_in_session += len(buffer_ids)

    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user. Current checkpoint state has been preserved.")
        print("You can rerun this command anytime to resume from the last saved checkpoint.")
    except Exception as e:
        logger.exception("Sparse embedding generation failed with exception")
        print(f"\n❌ Error during generation: {e}")
    finally:
        if manager:
            manager.close()

    total_elapsed = time.perf_counter() - overall_start
    final_completed = already_done + processed_in_session

    print("\n" + "=" * 80)
    print("SPARSE GENERATION SUMMARY")
    print("=" * 80)
    if target_total > 0:
        print(f"Total Target Points:       {target_total:,}")
    print(f"Processed in This Session: {processed_in_session:,}")
    print(f"Total Completed Checkpoint:{final_completed:,}")
    print(f"Total Session Time:        {total_elapsed / 60:.2f} minutes")
    if total_elapsed > 0 and processed_in_session > 0:
        print(f"Average Processing Rate:   {processed_in_session / total_elapsed:.1f} points/sec")
    print(f"Checkpoints Location:      {config.sparse_checkpoint_dir.resolve()}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate FastEmbed BM25 sparse vectors for Qdrant collection chunks with checkpointing."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Target Qdrant collection name (default: from config).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Optional input JSONL file containing {'id': ..., 'text': ...} chunks for offline or Colab generation.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qdrant/bm25",
        help="Sparse embedding model name (default: 'Qdrant/bm25').",
    )
    parser.add_argument(
        "--sparse-name",
        type=str,
        default="bm25",
        help="Sparse vector name in Qdrant (default: 'bm25').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding generation (default: 128).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save JSONL sparse vector checkpoints (default: data/sparse_checkpoints).",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        default=False,
        help="Disable writing JSONL checkpoint files to disk.",
    )
    parser.add_argument(
        "--apply-to-qdrant",
        action="store_true",
        default=False,
        help="Directly apply sparse vectors to Qdrant collection points (default: False, non-destructive).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Do not resume from existing checkpoints; process from start.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of points to process (useful for testing).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads for FastEmbed inference session.",
    )

    args = parser.parse_args()

    run_sparse_generation(
        collection_name=args.collection,
        input_file=args.input_file,
        model_name=args.model_name,
        sparse_name=args.sparse_name,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        save_checkpoints=not args.no_checkpoints,
        apply_to_qdrant=args.apply_to_qdrant,
        resume=not args.no_resume,
        sample_limit=args.limit,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
