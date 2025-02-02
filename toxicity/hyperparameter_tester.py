import asyncio
import time
import json
from datetime import datetime
import logging
from typing import Dict, List
from pathlib import Path
from toxicity.predict_v3 import OptimizedPipelineV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
GLOBAL_LIMIT = 1000
FETCH_SIZE = 1000


class HyperparameterTesterV2:
    """Test different hyperparameter configurations for the OptimizedPipelineV2."""

    def __init__(self, service_account_path: str = ".secrets/service-account.json"):
        """Initialize the tester."""
        self.service_account_path = Path(service_account_path)
        if not self.service_account_path.exists():
            raise FileNotFoundError(
                f"Service account file not found at {service_account_path}"
            )

        # "batch_size": 128,
        # "max_workers": 4,
        # "pipeline_batch_size": 16,
        # "max_concurrent_batches": 8

        self.param_grid = {
            "batch_size": [128, 64],
            "max_workers": [4, 8, 16],
            "pipeline_batch_size": [
                16,
                4,
                8,
                32,
            ],
            "max_concurrent_batches": [
                8,
                4,
                10,
                12,
                16,
                20,
            ],  # New parameter for controlling parallel batch processing
        }
        self.results: List[Dict] = []
        self.best_params: Dict = {}
        self.best_throughput: float = 0.0

    async def test_configuration(self, params: Dict) -> Dict:
        """Test a single parameter configuration."""
        try:
            pipeline = OptimizedPipelineV2(
                batch_size=params["batch_size"],
                max_workers=params["max_workers"],
                pipeline_batch_size=params["pipeline_batch_size"],
                max_concurrent_batches=params["max_concurrent_batches"],
                service_account_path=str(self.service_account_path),
            )

            pipeline.setup_table()

            start_time = time.perf_counter()
            await pipeline.process_tweets(
                fetch_size=FETCH_SIZE, total_limit=GLOBAL_LIMIT
            )
            duration = time.perf_counter() - start_time

            throughput = pipeline.total_processed / duration if duration > 0 else 0

            result = {
                "params": params,
                "throughput": throughput,
                "duration": duration,
                "total_processed": pipeline.total_processed,
                "timestamp": datetime.now().isoformat(),
                "batch_processing_time": (
                    duration / (pipeline.total_processed / params["batch_size"])
                    if pipeline.total_processed > 0
                    else 0
                ),
            }

            if throughput > self.best_throughput:
                self.best_throughput = throughput
                self.best_params = params.copy()

            return result

        except Exception as e:
            logger.error(f"Error testing configuration: {e}", exc_info=True)
            return {
                "params": params,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def save_results(self, intermediate: bool = False) -> None:
        """Save test results to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hyperparameter_results_v2{'_intermediate' if intermediate else ''}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "results": self.results,
                    "best_params": self.best_params,
                    "best_throughput": self.best_throughput,
                },
                f,
                indent=2,
            )

        if not intermediate:
            logger.info(f"Final results saved to {filename}")

    def print_summary(self) -> None:
        """Print a summary of the test results."""
        print("\n=== Hyperparameter Testing Summary (V2) ===")

        if self.best_params:
            print("\nBest configuration:")
            print(json.dumps(self.best_params, indent=2))
            print(f"Best throughput: {self.best_throughput:.2f} tweets/second")

        print("\nTop 5 configurations by throughput:")
        sorted_results = sorted(
            [r for r in self.results if "throughput" in r],
            key=lambda x: x["throughput"],
            reverse=True,
        )

        for idx, result in enumerate(sorted_results[:5]):
            print(f"\n{idx+1}. Configuration:")
            print(f"Params: {json.dumps(result['params'], indent=2)}")
            print(f"Throughput: {result['throughput']:.2f} tweets/second")
            print(f"Duration: {result['duration']:.2f} seconds")
            print(f"Total processed: {result['total_processed']} tweets")
            if "batch_processing_time" in result:
                print(
                    f"Average batch processing time: {result['batch_processing_time']:.2f} seconds"
                )

    async def run_tests(self) -> None:
        """Run all hyperparameter combinations."""
        from itertools import product

        combinations = [
            dict(zip(self.param_grid.keys(), v))
            for v in product(*self.param_grid.values())
        ]

        logger.info(f"Testing {len(combinations)} parameter combinations...")

        for i, params in enumerate(combinations, 1):
            logger.info(f"\nTesting configuration {i}/{len(combinations)}:")
            logger.info(json.dumps(params, indent=2))

            try:
                result = await self.test_configuration(params)
                self.results.append(result)

                if "throughput" in result:
                    logger.info(f"Throughput: {result['throughput']:.2f} tweets/second")
                    logger.info(f"Duration: {result['duration']:.2f} seconds")
                    logger.info(f"Total processed: {result['total_processed']} tweets")
                    if "batch_processing_time" in result:
                        logger.info(
                            f"Average batch processing time: {result['batch_processing_time']:.2f} seconds"
                        )
                else:
                    logger.error(f"Error: {result.get('error', 'Unknown error')}")

                # Save intermediate results after each configuration
                self.save_results(intermediate=True)

            except Exception as e:
                logger.error(f"Error running test configuration: {e}", exc_info=True)
                continue

            # Add a delay between tests to ensure clean state
            await asyncio.sleep(5)

        self.save_results()
        self.print_summary()


async def main() -> None:
    """Main entry point."""
    try:
        tester = HyperparameterTesterV2()
        await tester.run_tests()
    except FileNotFoundError as e:
        logger.error(f"Setup error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
