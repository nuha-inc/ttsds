from typing import List, Optional, Dict
import importlib.resources
from time import time
from pathlib import Path
import pickle
import gzip
import requests

import pandas as pd
from transformers import logging
import numpy as np
from sklearn.decomposition import PCA
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

from ttsds.benchmarks.benchmark import DeviceSupport
from ttsds.benchmarks.environment.voicefixer import VoiceFixerBenchmark
from ttsds.benchmarks.environment.wada_snr import WadaSNRBenchmark
from ttsds.benchmarks.general.hubert import HubertBenchmark
from ttsds.benchmarks.general.wav2vec2 import Wav2Vec2Benchmark
from ttsds.benchmarks.general.wavlm import WavLMBenchmark
from ttsds.benchmarks.intelligibility.w2v2_wer import Wav2Vec2WERBenchmark
from ttsds.benchmarks.intelligibility.w2v2_activations import Wav2Vec2ActivationsBenchmark
from ttsds.benchmarks.intelligibility.whisper_wer import WhisperWERBenchmark
from ttsds.benchmarks.intelligibility.whisper_activations import WhisperActivationsBenchmark
from ttsds.benchmarks.prosody.mpm import MPMBenchmark
from ttsds.benchmarks.prosody.pitch import PitchBenchmark
from ttsds.benchmarks.prosody.hubert_token import (
    HubertTokenBenchmark,
    HubertTokenSRBenchmark,
)
from ttsds.benchmarks.prosody.allosaurus import AllosaurusBenchmark
from ttsds.benchmarks.speaker.wespeaker import WeSpeakerBenchmark
from ttsds.benchmarks.speaker.dvector import DVectorBenchmark
from ttsds.benchmarks.benchmark import BenchmarkCategory
from ttsds.util.dataset import Dataset, DataDistribution

# we do this to avoid "some weights of the model checkpoint at ... were not used when initializing" warnings
logging.set_verbosity_error()


BENCHMARKS_V1 = {
    "hubert": HubertBenchmark,
    "wav2vec2": Wav2Vec2Benchmark,
    "wavlm": WavLMBenchmark,
    "wav2vec2_wer": Wav2Vec2WERBenchmark,
    "wav2vec2_activations": Wav2Vec2ActivationsBenchmark,
    "whisper_wer": WhisperWERBenchmark,
    # "whisper_activations": WhisperActivationsBenchmark,
    "mpm": MPMBenchmark,
    "pitch": PitchBenchmark,
    "wespeaker": WeSpeakerBenchmark,
    "dvector": DVectorBenchmark,
    "hubert_token": HubertTokenBenchmark,
    # "voicefixer": VoiceFixerBenchmark,
    "wada_snr": WadaSNRBenchmark,
}

BENCHMARKS_V2 = {
    "allosaurus": AllosaurusBenchmark,
    "hubert_token_sr": HubertTokenSRBenchmark,
}


class BenchmarkSuite:

    def __init__(
        self,
        datasets: List[Dataset],
        noise_datasets: List[Dataset],
        reference_datasets: List[Dataset],
        benchmarks: Dict[str, "Benchmark"] = BENCHMARKS_V1,
        print_results: bool = True,
        skip_errors: bool = False,
        write_to_file: str = None,
        multiprocessing: bool = False,
        n_processes: int = cpu_count(),
        benchmark_kwargs: dict = {},
        device: str = "cpu",
    ):
        # Extract bootstrap parameters from benchmark_kwargs
        self.bootstrap = benchmark_kwargs.get('bootstrap', False)
        self.n_bootstrap = benchmark_kwargs.get('n_bootstrap', 1000)
        self.confidence = benchmark_kwargs.get('confidence', 0.95)
        self.save_detailed = benchmark_kwargs.get('save_detailed', False)
        
        self.datasets = datasets
        self.benchmarks = benchmarks
        self.print_results = print_results
        self.skip_errors = skip_errors
        self.device = device
        
        # Initialize database if needed
        if write_to_file is not None and Path(write_to_file).exists():
            self.database = pd.read_csv(write_to_file)
        else:
            self.database = pd.DataFrame()
        
        self.datasets = sorted(self.datasets, key=lambda x: x.name)
        self.database = pd.DataFrame(
            columns=[
                "benchmark_name",
                "benchmark_category",
                "dataset",
                "score",
                "ci",
                "time_taken",
                "noise_dataset",
                "reference_dataset",
            ]
        )
        self.noise_datasets = noise_datasets
        self.reference_datasets = reference_datasets
        self.multiprocessing = multiprocessing
        self.n_processes = n_processes
        self.file_results = {}  # Store file-level results

        # Initialize benchmarks with proper parameters
        benchmark_init_kwargs = benchmark_kwargs.copy()
        if "hubert_token" not in benchmark_init_kwargs:
            benchmark_init_kwargs["hubert_token"] = {}
        if "cluster_datasets" not in benchmark_init_kwargs["hubert_token"]:
            benchmark_init_kwargs["hubert_token"]["cluster_datasets"] = [
                reference_datasets[0].sample(min(100, len(reference_datasets[0])))
            ]
        
        if "hubert_token_sr" not in benchmark_init_kwargs:
            benchmark_init_kwargs["hubert_token_sr"] = {}
        if "cluster_datasets" not in benchmark_init_kwargs["hubert_token_sr"]:
            benchmark_init_kwargs["hubert_token_sr"]["cluster_datasets"] = [
                reference_datasets[0].sample(min(100, len(reference_datasets[0])))
            ]

        self.benchmarks = {
            k: v(**benchmark_init_kwargs.get(k, {}))
            for k, v in benchmarks.items()
        }

        # Move benchmarks to GPU if supported
        for benchmark in self.benchmarks.values():
            if DeviceSupport.GPU in benchmark.supported_devices and device == "cuda":
                benchmark.to_device(device)

        if self.multiprocessing:
            if len(benchmarks) > len(noise_datasets) + len(reference_datasets):
                print(f"Running benchmarks without multiprocessing")
                self.noise_distributions = [
                    DataDistribution(
                        ds,
                        benchmarks=self.benchmarks,
                        name=f"speech_{ds.name}",
                        multiprocessing=self.multiprocessing,
                        n_processes=self.n_processes,
                    )
                    for ds in noise_datasets
                ]
                self.reference_distributions = [
                    DataDistribution(
                        ds,
                        benchmarks=self.benchmarks,
                        name=ds.name,
                        multiprocessing=self.multiprocessing,
                        n_processes=self.n_processes,
                        cache_distributions=not self.bootstrap  # Disable caching if bootstrapping
                    )
                    for ds in reference_datasets
                ]
            else:
                print(f"Running benchmarks with multiprocessing with {self.n_processes} processes and {len(noise_datasets) + len(reference_datasets)} datasets")
                tasks = [d for d in noise_datasets + reference_datasets]
                with ThreadPoolExecutor(max_workers=self.n_processes) as executor:
                    results = list(
                        executor.map(
                            lambda x: self.get_data_distribution(x, self.benchmarks, is_reference=x in reference_datasets),
                            tasks,
                        )
                    )
                self.noise_distributions = results[: len(noise_datasets)]
                self.reference_distributions = results[len(noise_datasets) :]
        else:
            self.noise_distributions = [
                self.get_data_distribution(ds, self.benchmarks)
                for ds in noise_datasets
            ]
            self.reference_distributions = [
                self.get_data_distribution(ds, self.benchmarks, is_reference=True)
                for ds in reference_datasets
            ]
        self.write_to_file = write_to_file
        if Path(write_to_file).exists():
            self.database = pd.read_csv(write_to_file, index_col=0)
            self.database = self.database.reset_index()

    def get_data_distribution(self, dataset: Dataset, benchmarks: Dict[str, "Benchmark"], is_reference: bool = False) -> DataDistribution:
        return DataDistribution(
            dataset,
            benchmarks=benchmarks,
            name=dataset.name if not dataset.name.startswith("speech_") else f"speech_{dataset.name}",
            multiprocessing=False,
            cache_distributions=not (self.bootstrap and is_reference)  # Disable caching for reference datasets if bootstrapping
        )

    def _get_distribution(self, benchmark: "Benchmark", dataset: Dataset) -> np.ndarray:
        return benchmark.get_distribution(dataset)

    def _run_benchmark(self, benchmark: "Benchmark", dataset: Dataset) -> dict:
        print("\n")
        print(f"{'='*80}")
        print(f"Benchmark Category: {benchmark.category.name}")
        if hasattr(dataset, "root_dir"):
            print(f"Running {benchmark.name} on {dataset.root_dir}")
        else:
            print(f"Running {benchmark.name} on {dataset.name}")
        try:
            start = time()
            # Pass bootstrap parameters to compute_score
            score, ci, datasets, file_results = benchmark.compute_score(
                dataset, 
                self.reference_distributions, 
                self.noise_distributions,
                n_bootstrap=self.n_bootstrap if self.bootstrap else 0,
                confidence=self.confidence
            )
            time_taken = time() - start
            
            # Store file results if enabled
            if self.save_detailed and file_results:
                if dataset.name not in self.file_results:
                    self.file_results[dataset.name] = {}
                self.file_results[dataset.name][benchmark.name] = file_results
                
        except Exception as e:
            if self.skip_errors:
                print(f"Error: {e}")
                score = np.nan
                ci = (0.0, 0.0)
                time_taken = np.nan
                datasets = (None, None)
            else:
                raise e

        result = {
            "benchmark_name": [benchmark.name],
            "benchmark_category": [benchmark.category.value],
            "dataset": [dataset.name],
            "score": [score],
            "ci": [f"{ci[0]:.6f},{ci[1]:.6f}" if isinstance(ci, tuple) else "0.0,0.0"],
            "time_taken": [time_taken],
            "noise_dataset": [datasets[0]],
            "reference_dataset": [datasets[1]],
        }
        return result

    def run(self) -> pd.DataFrame:
        tasks = []
        for benchmark in sorted(self.benchmarks.values(), key=lambda x: x.name):
            for dataset in self.datasets:
                if (
                    (self.database["benchmark_name"] == benchmark.name)
                    & (self.database["dataset"] == dataset.name)
                ).any():
                    print(
                        f"Skipping {benchmark.name} on {dataset.name} as it's already in the database"
                    )
                    continue
                tasks.append((benchmark, dataset))
        if self.multiprocessing:
            print(f"Running benchmarks with {self.n_processes} processes")
            with ThreadPoolExecutor(max_workers=self.n_processes) as executor:
                results = list(executor.map(lambda x: self._get_distribution(*x), tasks))
        results = []
        if not self.multiprocessing:
            for benchmark, dataset in tasks:
                result = self._run_benchmark(benchmark, dataset)
                results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=self.n_processes) as executor:
                results = list(executor.map(lambda x: self._run_benchmark(*x), tasks))
        for result in results:
            if self.print_results:
                print(result)
            # Convert confidence interval tuple to string for storage
            if isinstance(result['ci'][0], tuple):
                result['ci'] = [f"{result['ci'][0][0]:.6f},{result['ci'][0][1]:.6f}"]
            self.database = pd.concat(
                [
                    self.database,
                    pd.DataFrame(result),
                ],
                ignore_index=True,
            )
            if self.write_to_file is not None:
                self.database["score"] = self.database["score"].astype(float)
                # Convert stored CI string back to tuple when needed
                if 'ci' in self.database.columns:
                    def parse_ci(ci_str):
                        if isinstance(ci_str, str) and ',' in ci_str:
                            width, conf = ci_str.split(',')
                            return (float(width), float(conf))
                        return (0.0, 0.0)
                    self.database["ci"] = self.database["ci"].apply(parse_ci)
                self.database = self.database.sort_values(
                    ["benchmark_category", "benchmark_name", "score"],
                    ascending=[True, True, False],
                )
                self.database.to_csv(self.write_to_file, index=False)
        return self.database

    @staticmethod
    def aggregate_df(df: pd.DataFrame) -> pd.DataFrame:
        def concat_text(x):
            return ", ".join(x)

        df["benchmark_category"] = df["benchmark_category"].apply(
            lambda x: BenchmarkCategory(x).name
        )
        df = (
            df.groupby(
                [
                    "benchmark_category",
                    "dataset",
                ]
            )
            .agg(
                {
                    "score": ["mean"],
                    "ci": ["mean"],
                    "time_taken": ["mean"],
                    "noise_dataset": [concat_text],
                    "reference_dataset": [concat_text],
                    "benchmark_name": [concat_text],
                }
            )
            .reset_index()
        )
        # remove multiindex
        df.columns = [x[0] for x in df.columns.ravel()]
        # drop the benchmark_name column
        df = df.drop("benchmark_name", axis=1)
        # replace benchmark_category number with string
        return df

    def get_aggregated_results(self) -> pd.DataFrame:
        df = self.database.copy()
        return BenchmarkSuite.aggregate_df(df)

    def get_benchmark_distribution(
        self,
        benchmark_name: str,
        dataset_name: str,
        pca_components: Optional[int] = None,
    ) -> dict:
        benchmark = [x for x in self.benchmarks.values() if x.name == benchmark_name][0]
        dataset = [x for x in self.datasets if x.name == dataset_name][0]
        closest_noise = self.database[
            (self.database["benchmark_name"] == benchmark_name)
            & (self.database["dataset"] == dataset_name)
        ]["noise_dataset"].values[0]
        closest_noise = [
            x for x in self.noise_distributions if x.name == closest_noise
        ][0]
        other_noise = [
            x for x in self.noise_distributions if x.name != closest_noise.name
        ][0]
        closest_reference = self.database[
            (self.database["benchmark_name"] == benchmark_name)
            & (self.database["dataset"] == dataset_name)
        ]["reference_dataset"].values[0]
        closest_reference = [
            x for x in self.reference_distributions if x.name == closest_reference
        ][0]
        other_reference = [
            x for x in self.reference_distributions if x.name != closest_reference.name
        ][0]
        result = {
            "benchmark_distribution": benchmark.get_distribution(dataset),
            "noise_distribution": benchmark.get_distribution(closest_noise),
            "reference_distribution": benchmark.get_distribution(closest_reference),
            "other_noise_distribution": benchmark.get_distribution(other_noise),
            "other_reference_distribution": benchmark.get_distribution(other_reference),
        }
        if pca_components is not None:
            pca = PCA(n_components=pca_components)
            # fit on all except the benchmark distribution
            pca.fit(
                np.vstack(
                    [v for k, v in result.items() if k != "benchmark_distribution"]
                )
            )
            result = {k: pca.transform(v) for k, v in result.items()}
        return result