import asyncio
import time
from datetime import datetime
import torch
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
import logging
from typing import List, Dict, Tuple, Optional
from bigquery_client import BigQueryClient
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIMIT = None
FETCH_SIZE = 5120
CREATE_TABLE = False
TOXICITY_MODEL = "textdetox/xlmr-large-toxicity-classifier"
SENTIMENT_MODEL = "oliverguhr/german-sentiment-bert"


class OptimizedPipeline:
    """Pipeline for processing tweets with parallel batch processing."""

    def __init__(
        self,
        batch_size: int = 128,
        max_workers: int = 16,
        pipeline_batch_size: int = 16,
        max_concurrent_batches: int = 8,
        service_account_path: str = ".secrets/service-account.json",
    ):
        self.project_id = "grounded-nebula-408412"
        self.dataset_id = "twitter_analysis_curated"
        self.filter_table = "selected_hashtag_topic_count"
        self.source_table = "relevant_tweets"
        self.target_table = "tweet_sentiment_analysis"
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_workers = max_workers
        self.pipeline_batch_size = pipeline_batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.service_account_path = service_account_path

        self.batch_semaphore = asyncio.Semaphore(max_concurrent_batches)
        self.bq_client = BigQueryClient(self.project_id,
                                        self.service_account_path)

        self.sentiment_pipeline = None
        self.toxicity_pipeline = None

        # Cumulative statistics for timing
        self.total_inference_time = 0
        self.total_batch_time = 0
        self.total_queue_time = 0
        self.total_batches = 0

        self.total_processed = 0
        self.processing_start_time = None

        self.bq_queue = asyncio.Queue()
        self.streaming_task = None

    def setup_table(self) -> None:
        """Create BigQuery table and ensure it's ready."""
        schema = [
            bigquery.SchemaField("tweet_id", "STRING"),
            bigquery.SchemaField("user_id", "STRING"),
            bigquery.SchemaField("recorded_at", "TIMESTAMP"),
            bigquery.SchemaField("text", "STRING"),
            bigquery.SchemaField("sentiment", "STRING"),
            bigquery.SchemaField("positive_probability", "FLOAT"),
            bigquery.SchemaField("neutral_probability", "FLOAT"),
            bigquery.SchemaField("negative_probability", "FLOAT"),
            bigquery.SchemaField("toxicity_label", "STRING"),
            bigquery.SchemaField("toxicity_score", "FLOAT"),
        ]

        if CREATE_TABLE is True:
            self.bq_client.create_table(self.dataset_id, self.target_table,
                                        schema)

        self.init_models()

    def init_models(self) -> None:
        """Initialize the transformer pipelines."""
        device = 0 if torch.cuda.is_available() else -1

        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=device,
            batch_size=self.pipeline_batch_size,
            truncation=True,
            max_length=512,
        )

        self.toxicity_pipeline = pipeline(
            "text-classification",
            model=TOXICITY_MODEL,
            device=device,
            batch_size=self.pipeline_batch_size,
            truncation=True,
            max_length=512,
        )

    async def parallel_inference(self, texts: List[str]) -> Tuple[List, List]:
        """Run sentiment and toxicity analysis in parallel."""
        loop = asyncio.get_running_loop()
        inference_start = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            sentiment_future = loop.run_in_executor(executor,
                                                    self.sentiment_pipeline,
                                                    texts)
            toxicity_future = loop.run_in_executor(executor,
                                                   self.toxicity_pipeline,
                                                   texts)

            sentiment_outputs, toxicity_outputs = await asyncio.gather(
                sentiment_future, toxicity_future)

        # Update cumulative inference time
        self.total_inference_time += time.time() - inference_start

        return sentiment_outputs, toxicity_outputs

    async def process_batch(self, batch: List[Dict]) -> None:
        """Process a single batch of tweets."""
        async with self.batch_semaphore:
            batch_start = time.time()

            try:
                texts = [item["text"] for item in batch]
                sentiment_outputs, toxicity_outputs = await self.parallel_inference(
                    texts)

                full_results = []
                for item, sent, tox in zip(batch, sentiment_outputs,
                                           toxicity_outputs):
                    result = {
                        "tweet_id":
                        item["tweet_id"],
                        "user_id":
                        item["user_id"],
                        "recorded_at":
                        item["recorded_at"],
                        "text":
                        item["text"],
                        "sentiment":
                        sent["label"],
                        "positive_probability":
                        float(sent["score"] if sent["label"] ==
                              "positive" else 1 - sent["score"]),
                        "neutral_probability":
                        float(sent["score"] if sent["label"] ==
                              "neutral" else 1 - sent["score"]),
                        "negative_probability":
                        float(sent["score"] if sent["label"] ==
                              "negative" else 1 - sent["score"]),
                        "toxicity_label":
                        tox["label"],
                        "toxicity_score":
                        float(tox["score"]),
                    }
                    full_results.append(result)

                queue_start = time.time()
                await self.bq_queue.put(full_results)
                self.total_processed += len(full_results)

                # Update cumulative queue and batch time
                self.total_queue_time += time.time() - queue_start
                self.total_batch_time += time.time() - batch_start
                self.total_batches += 1

                logger.info(f"Batch processed: {len(full_results)} tweets. "
                            f"Total processed: {self.total_processed}")

            except Exception as e:
                logger.error(f"Error processing batch: {e}", exc_info=True)

    async def stream_to_bigquery(self) -> None:
        """Background task to stream results to BigQuery."""
        while True:
            try:
                rows = await self.bq_queue.get()

                if rows == "DONE":
                    logger.info(
                        "Received DONE signal, finishing streaming task")
                    self.bq_queue.task_done()
                    break

                try:
                    for row in rows:
                        if isinstance(row.get("recorded_at"), datetime):
                            row["recorded_at"] = row["recorded_at"].isoformat()

                    self.bq_client.insert_rows_json(self.dataset_id,
                                                    self.target_table, rows)
                    logger.info(
                        f"Successfully streamed {len(rows)} rows to BigQuery")

                except Exception as e:
                    logger.error(f"Error streaming to BigQuery: {e}",
                                 exc_info=True)

                finally:
                    self.bq_queue.task_done()

            except Exception as e:
                logger.error(f"Fatal error in streaming task: {e}",
                             exc_info=True)
                self.bq_queue.task_done()
                break

    async def process_tweets(
        self,
        fetch_size: int = 10000,
        total_limit: Optional[int] = None,
    ) -> None:
        """Process tweets with parallel batch processing."""
        if not self.sentiment_pipeline or not self.toxicity_pipeline:
            self.init_models()

        self.streaming_task = asyncio.create_task(self.stream_to_bigquery())
        self.processing_start_time = time.perf_counter()

        processed_tweets = 0
        total_processed = 0

        try:
            while True:
                if total_limit and total_processed >= total_limit:
                    logger.info(
                        f"Reached processing limit of {total_limit} tweets")
                    break

                current_fetch = fetch_size
                if total_limit:
                    remaining = total_limit - total_processed
                    current_fetch = min(fetch_size, remaining)

                query = f"""
                    SELECT tweet_id, user_id, text, recorded_at
                    FROM `{self.project_id}.{self.dataset_id}.{self.source_table}`
                    WHERE text IS NOT NULL AND LENGTH(TRIM(text)) > 0
                    AND tweet_id NOT IN (SELECT tweet_id FROM `{self.project_id}.{self.dataset_id}.{self.target_table}`)
                    ORDER BY recorded_at ASC
                    LIMIT {current_fetch}
                """

                try:
                    rows = self.bq_client.query(query)
                    if not len(rows):
                        logger.info(
                            f"No more rows to process after {total_processed} tweets"
                        )
                        break

                    logger.info(
                        f"Retrieved {len(rows)} rows from BigQuery (processed_tweets {processed_tweets})"
                    )

                    batches = []
                    current_batch = []

                    for _, row in rows.iterrows():
                        if row.text and row.text.strip():
                            current_batch.append({
                                "text":
                                row.text,
                                "tweet_id":
                                row.tweet_id,
                                "user_id":
                                row.user_id,
                                "recorded_at":
                                row.recorded_at,
                            })
                            if len(current_batch) >= self.batch_size:
                                batches.append(current_batch)
                                current_batch = []

                    if current_batch:
                        batches.append(current_batch)

                    batch_tasks = [
                        self.process_batch(batch) for batch in batches
                    ]
                    await asyncio.gather(*batch_tasks)
                    await self.bq_queue.join()

                    total_processed += len(rows)
                    processed_tweets += len(rows)

                    logger.info(f"Processed {total_processed} tweets" + (
                        f" out of {total_limit}" if total_limit else ""))

                except Exception as e:
                    logger.error(f"Error in process_tweets: {e}",
                                 exc_info=True)
                    raise

            await self.bq_queue.join()
            await self.bq_queue.put("DONE")
            await self.streaming_task

        except Exception as e:
            logger.error(f"Fatal error in process_tweets: {e}", exc_info=True)
            raise

        finally:
            if self.streaming_task and not self.streaming_task.done():
                self.streaming_task.cancel()
                try:
                    await self.streaming_task
                except asyncio.CancelledError:
                    pass

            elapsed = time.perf_counter() - self.processing_start_time
            avg_inference_time = (self.total_inference_time /
                                  self.total_batches
                                  if self.total_batches > 0 else 0)
            avg_batch_time = (self.total_batch_time / self.total_batches
                              if self.total_batches > 0 else 0)
            avg_queue_time = (self.total_queue_time / self.total_batches
                              if self.total_batches > 0 else 0)

            logger.info(
                f"\nFinal Pipeline Statistics:"
                f"\n- Total time: {elapsed:.2f}s"
                f"\n- Total processed: {self.total_processed} tweets"
                f"\n- Overall throughput: {self.total_processed/elapsed:.2f} tweets/second"
                f"\n- Average inference time per batch: {avg_inference_time:.2f}s"
                f"\n- Average total batch time: {avg_batch_time:.2f}s"
                f"\n- Average queue time: {avg_queue_time:.2f}s")


async def main() -> None:
    """Main entry point."""
    try:
        processor = OptimizedPipelineV2()
        processor.setup_table()
        await processor.process_tweets(fetch_size=FETCH_SIZE,
                                       total_limit=LIMIT)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
