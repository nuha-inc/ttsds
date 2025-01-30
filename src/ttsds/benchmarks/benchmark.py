"""
This file contains the Benchmark abstract class.
"""

from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import importlib.resources
import json
from typing import List, Union, Optional, Tuple, Dict
from functools import lru_cache
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from ttsds.util.dataset import Dataset, DataDistribution
from ttsds.util.cache import cache, load_cache, check_cache, hash_md5
from ttsds.util.distances import wasserstein_distance, frechet_distance


class BenchmarkCategory(Enum):
    """
    Enum class for the different categories of benchmarks.
    """

    OVERALL = 1
    PROSODY = 2
    ENVIRONMENT = 3
    SPEAKER = 4
    PHONETICS = 5
    INTELLIGIBILITY = 6
    TRAINABILITY = 7
    EXTERNAL = 8


class BenchmarkDimension(Enum):
    """
    Enum class for the different dimensions of benchmarks.
    """

    ONE_DIMENSIONAL = 1
    N_DIMENSIONAL = 2

class DeviceSupport(Enum):
    """
    Enum class for the different device support of benchmarks.
    """

    CPU = 1
    GPU = 2


class Benchmark(ABC):
    """
    Abstract class for a benchmark.
    """

    def __init__(
        self,
        name: str,
        category: BenchmarkCategory,
        dimension: BenchmarkDimension,
        description: str,
        version: Optional[str] = None,
        supported_devices: List[DeviceSupport] = [DeviceSupport.CPU],
        **kwargs,
    ):
        self.name = name
        self.key = name.lower().replace(" ", "_")
        self.category = category
        self.dimension = dimension
        self.description = description
        self.version = version
        self.kwargs = kwargs
        self.supported_devices = supported_devices

    def get_distribution(self, dataset: Union[Dataset, DataDistribution]) -> np.ndarray:
        """
        Abstract method to get the distribution of the benchmark.
        If the benchmark is one-dimensional, the method should return a
        numpy array with the values of the benchmark for each sample in the dataset.
        If the benchmark is n-dimensional, the method should return a numpy array
        with the values of the benchmark for each sample in the dataset, where each
        row corresponds to a sample and each column corresponds to a dimension of the benchmark.
        """
        ds_hash = hash_md5(dataset)
        benchmark_hash = hash_md5(self)
        cache_name = f"benchmarks/{self.name}/{ds_hash}_{benchmark_hash}"
        if check_cache(cache_name):
            result = load_cache(cache_name)
            if result is not None:
                return result
        if check_cache(cache_name + "_mu") and check_cache(cache_name + "_sig"):
            mu = load_cache(cache_name + "_mu")
            sig = load_cache(cache_name + "_sig")
            if mu is not None and sig is not None:
                return (mu, sig)
        if (
            isinstance(dataset, DataDistribution)
            and self.dimension == BenchmarkDimension.N_DIMENSIONAL
        ):
            mu, sig = dataset.get_distribution(self.key)
            cache(mu, cache_name + "_mu")
            cache(sig, cache_name + "_sig")
            return (mu, sig)
        elif (
            isinstance(dataset, DataDistribution)
            and self.dimension == BenchmarkDimension.ONE_DIMENSIONAL
        ):
            distribution = dataset.get_distribution(self.key)
            cache(distribution, cache_name)
            return distribution
        distribution = self._get_distribution(dataset)
        cache(distribution, cache_name)
        return distribution

    @abstractmethod
    def _get_distribution(self, dataset: Dataset) -> np.ndarray:
        """
        Abstract method to get the distribution of the benchmark.
        """
        raise NotImplementedError

    def to_device(self, device: str):
        """
        Move the benchmark to a device.
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError("Invalid device")
        if self.supported_devices == [DeviceSupport.CPU]:
            if device == "cuda":
                raise ValueError("Benchmark does not support CUDA")
        self._to_device(device)

    def _to_device(self, device: str):
        """
        Abstract method to move the benchmark to a device.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.category.name}/{self.name}"

    def __repr__(self):
        return f"{self.category.name}/{self.name}"

    def __hash__(self) -> int:
        h = hashlib.md5()
        h.update(self.name.encode())
        h.update(self.category.name.encode())
        h.update(self.dimension.name.encode())
        h.update(self.description.encode())
        if self.version is not None:
            h.update(self.version.encode())
        # convert the kwargs to strings
        kwargs_str = {
            k: str(v) if not isinstance(v, dict) else json.dumps(v, sort_keys=True)
            for k, v in self.kwargs.items()
        }
        h.update(json.dumps(kwargs_str, sort_keys=True).encode())
        return int(h.hexdigest(), 16)

    def compute_distance(self, dataset1: Dataset, dataset2: Dataset) -> float:
        """
        Compute the distance between two datasets.
        """
        dist1 = self.get_distribution(dataset1)
        dist2 = self.get_distribution(dataset2)
        
        # Store per-file results for dataset2 (test dataset)
        self.file_results = {}
        for i, file_path in enumerate(dataset2.get_files()):
            try:
                file_score = self._compute_file_score(dist2[i] if isinstance(dist2, np.ndarray) else dist2, dist1)
                self.file_results[str(file_path)] = {
                    'score': float(file_score),
                    'benchmark': self.name,
                    'category': self.category.name
                }
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                self.file_results[str(file_path)] = {
                    'score': None,
                    'error': str(e),
                    'benchmark': self.name,
                    'category': self.category.name
                }
        
        # Compute overall distance based on dimension type
        if self.dimension == BenchmarkDimension.ONE_DIMENSIONAL:
            return wasserstein_distance(dist1, dist2)
        else:  # N_DIMENSIONAL
            # Handle both direct arrays and (mu, sigma) tuples
            if isinstance(dist1, tuple) and isinstance(dist2, tuple):
                # For (mu, sigma) tuples, use Frechet distance directly
                return frechet_distance(dist1[0], dist2[0], dist1[1], dist2[1])
            else:
                # For direct arrays, ensure proper shape
                if isinstance(dist1, np.ndarray) and len(dist1.shape) == 1:
                    dist1 = dist1.reshape(1, -1)
                if isinstance(dist2, np.ndarray) and len(dist2.shape) == 1:
                    dist2 = dist2.reshape(1, -1)
                return frechet_distance(dist1, dist2)

    def _compute_file_score(self, file_dist, reference_dist) -> float:
        """
        Compute score for a single file against reference distribution.
        """
        if self.dimension == BenchmarkDimension.ONE_DIMENSIONAL:
            return wasserstein_distance(np.array([file_dist]), reference_dist)
        else:  # N_DIMENSIONAL
            # Handle both direct arrays and (mu, sigma) tuples
            if isinstance(reference_dist, tuple):
                # For (mu, sigma) tuples, compute distance to the mean
                if isinstance(file_dist, tuple):
                    return frechet_distance(file_dist[0], reference_dist[0], file_dist[1], reference_dist[1])
                else:
                    # If file_dist is a direct array, treat it as a single point
                    if len(file_dist.shape) == 1:
                        file_dist = file_dist.reshape(1, -1)
                    return np.linalg.norm(file_dist.mean(axis=0) - reference_dist[0])
            else:
                # For direct arrays, use standard distance
                if len(file_dist.shape) == 1:
                    file_dist = file_dist.reshape(1, -1)
                if len(reference_dist.shape) == 1:
                    reference_dist = reference_dist.reshape(1, -1)
                return frechet_distance(file_dist, reference_dist)

    def compute_score(
        self,
        dataset: Dataset,
        reference_datasets: List[Dataset],
        noise_datasets: List[Dataset],
    ) -> Tuple[float, float, Tuple[str, str], Dict]:
        """
        Compute the score of the benchmark on a dataset.
        Now also returns per-file results.
        """
        noise_scores = []
        noise_file_results = []
        for noise_ds in noise_datasets:
            score = self.compute_distance(noise_ds, dataset)
            noise_scores.append(score)
            if hasattr(self, 'file_results'):
                noise_file_results.append(self.file_results)
        noise_scores = np.array(noise_scores)

        dataset_scores = []
        dataset_file_results = []
        for ref_ds in reference_datasets:
            score = self.compute_distance(ref_ds, dataset)
            dataset_scores.append(score)
            if hasattr(self, 'file_results'):
                dataset_file_results.append(self.file_results)
        dataset_scores = np.array(dataset_scores)

        closest_noise_idx = np.argmin(noise_scores)
        closest_dataset_idx = np.argmin(dataset_scores)

        print(f"Closest noise dataset: {noise_datasets[closest_noise_idx].name}")
        print(
            f"Closest reference dataset: {reference_datasets[closest_dataset_idx].name}"
        )

        noise_score = np.min(noise_scores)
        dataset_score = np.min(dataset_scores)

        combined_score = dataset_score + noise_score
        score = (noise_score / combined_score) * 100
        
        # Return the file results from the closest reference dataset
        file_results = dataset_file_results[closest_dataset_idx] if dataset_file_results else {}
        
        return (
            score,
            1.0,
            (
                noise_datasets[closest_noise_idx].name,
                reference_datasets[closest_dataset_idx].name,
            ),
            file_results
        )
