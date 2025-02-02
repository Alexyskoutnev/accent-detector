import logging
from pathlib import Path
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from inference.inference_engine import InferenceEngine
from evaluation.accent_evaluator import (
    AccentEvaluator,
    ModelEvaluationHelpers,
    EvalStats,
)
from preprocessors.audio_data_preprocessor import AudioDataPreprocessor
from storage.bucket_dao import BucketDAO


class AccentDetectionPipeline:
    """_summary_
    """

    def __init__(self, 
                 bucket_dao: BucketDAO, 
                 output_dir: str = "results"):
        self.bucket_dao = bucket_dao
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inference_engine = InferenceEngine()
        self.stats_history = []
        AccentEvaluator.reset_stats()

    def process_batch(
        self, batch_tensor: torch.Tensor, batch_labels: pd.DataFrame
    ) -> tuple[EvalStats, EvalStats]:
        try:
            batch_predictions = next(self.inference_engine.predict(batch_tensor))
            batch_stats, running_stats = AccentEvaluator.evaluate(
                batch_predictions, batch_labels
            )
            self.stats_history.append(running_stats)
            return batch_stats, running_stats
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            raise

    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_dir / f"metrics_history_{timestamp}.png"
        ModelEvaluationHelpers.plot_metrics_history(
            self.stats_history, save_path=str(plot_path)
        )
        matrix_path = self.output_dir / f"confusion_matrix_{timestamp}.png"
        ModelEvaluationHelpers.plot_confusion_matrix(
            self.stats_history[-1], save_path=str(matrix_path)
        )
        metrics_path = self.output_dir / f"metrics_{timestamp}.json"
        self.stats_history[-1].save_json(str(metrics_path))

    def run(self, running_in_notebook: bool = False):
        logging.info("Starting accent detection pipeline")
        try:
            for batch_num, (batch_tensor, batch_labels) in enumerate(
                tqdm(
                    AudioDataPreprocessor.process(self.bucket_dao),
                    desc="Processing batches",
                ),
                1,
            ):
                batch_stats, running_stats = self.process_batch(
                    batch_tensor, batch_labels
                )
                logging.info(
                    f"\nBatch {batch_num} Results:\n{batch_stats}\nRunning Statistics:\n{running_stats}"
                )
                if running_in_notebook:
                    plt.clf()
                    ModelEvaluationHelpers.plot_metrics_history(self.stats_history)
                    plt.pause(0.1)
        except Exception as e:
            logging.error(f"Pipeline error: {e}")
            raise
        finally:
            self.save_results()
            if running_in_notebook:
                plt.close()
            logging.info(f"Pipeline completed. Results saved to: {self.output_dir}")


if __name__ == "__main__":
    # Run the pipeline via command line
    from storage.mock_bucket import MockBucket
    logging.basicConfig(level=logging.INFO)
    bucket = MockBucket("./mock_bucket")
    dao = BucketDAO(bucket)
    pipeline = AccentDetectionPipeline(dao)
    pipeline.run()
