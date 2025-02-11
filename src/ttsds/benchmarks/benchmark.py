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

    def compute_distance(self, dataset1: Union[Dataset, DataDistribution], dataset2: Union[Dataset, DataDistribution], n_bootstrap: int = 1000, confidence: float = 0.95) -> Union[float, Tuple[float, float, float]]:
        """
        Compute the distance between two datasets with optional bootstrapping.
        
        Args:
            dataset1: Reference dataset or distribution
            dataset2: Test dataset or distribution
            n_bootstrap: Number of bootstrap samples for reference dataset (default: 1000)
            confidence: Confidence level for intervals (default: 0.95)
            
        Returns:
            If n_bootstrap > 0: Tuple of (mean_distance, lower_ci, upper_ci)
            If n_bootstrap = 0: Single distance value
        """
        print(f"\nDEBUG: Computing distance with n_bootstrap={n_bootstrap}, confidence={confidence}")
        
        # Get distribution for test dataset (no bootstrapping needed)
        if isinstance(dataset2, DataDistribution):
            dist2 = dataset2.get_distribution(self.key)
        else:
            dist2 = self.get_distribution(dataset2)
        
        # Store per-file results for dataset2 (test dataset)
        self.file_results = {}
        
        if n_bootstrap > 0:
            print(f"DEBUG: Starting bootstrap process")
            # Get base distribution for reference dataset
            if isinstance(dataset1, DataDistribution):
                print(f"DEBUG: Getting base distribution from DataDistribution")
                # Get the underlying dataset for bootstrapping
                base_dataset = dataset1.get_dataset()
                base_dist1 = self.get_distribution(base_dataset)
            else:
                print(f"DEBUG: Getting base distribution from Dataset")
                base_dist1 = self.get_distribution(dataset1)
            
            print(f"DEBUG: Base distribution shape: {base_dist1.shape if hasattr(base_dist1, 'shape') else 'tuple'}")
            
            # Perform bootstrap sampling
            bootstrap_distances = []
            for i in range(n_bootstrap):
                if i % 100 == 0:
                    print(f"DEBUG: Bootstrap iteration {i}/{n_bootstrap}")
                if self.dimension == BenchmarkDimension.ONE_DIMENSIONAL:
                    # Sample with replacement from the reference distribution
                    bootstrap_sample = np.random.choice(base_dist1, size=len(base_dist1), replace=True)
                    distance = wasserstein_distance(bootstrap_sample, dist2)
                else:  # N_DIMENSIONAL
                    # Sample with replacement from the reference embeddings
                    indices = np.random.choice(len(base_dist1), size=len(base_dist1), replace=True)
                    bootstrap_sample = base_dist1[indices]
                    if isinstance(bootstrap_sample, tuple):
                        mu = np.mean(bootstrap_sample[0], axis=0)
                        sigma = np.cov(bootstrap_sample[0], rowvar=False)
                        bootstrap_sample = (mu, sigma)
                    distance = frechet_distance(bootstrap_sample, dist2)
                bootstrap_distances.append(distance)
            
            # Calculate confidence intervals
            mean_distance = np.mean(bootstrap_distances)
            lower_percentile = (1 - confidence) / 2
            upper_percentile = 1 - lower_percentile
            ci_lower = np.percentile(bootstrap_distances, lower_percentile * 100)
            ci_upper = np.percentile(bootstrap_distances, upper_percentile * 100)
            
            print(f"DEBUG: Bootstrap results - mean: {mean_distance:.4f}, CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Store per-file results with confidence intervals
            if isinstance(dataset2, Dataset):
                files = dataset2.get_files()
            elif isinstance(dataset2, DataDistribution):
                files = dataset2.get_dataset().get_files()
            else:
                files = []
            
            for i, file_path in enumerate(files):
                try:
                    file_scores = []
                    for _ in range(n_bootstrap):
                        if self.dimension == BenchmarkDimension.ONE_DIMENSIONAL:
                            bootstrap_sample = np.random.choice(base_dist1, size=len(base_dist1), replace=True)
                            file_score = self._compute_file_score(dist2[i] if isinstance(dist2, np.ndarray) else dist2, bootstrap_sample)
                        else:
                            indices = np.random.choice(len(base_dist1), size=len(base_dist1), replace=True)
                            bootstrap_sample = base_dist1[indices]
                            file_score = self._compute_file_score(dist2[i] if isinstance(dist2, np.ndarray) else dist2, bootstrap_sample)
                        file_scores.append(file_score)
                    
                    mean_score = np.mean(file_scores)
                    score_ci_lower = np.percentile(file_scores, lower_percentile * 100)
                    score_ci_upper = np.percentile(file_scores, upper_percentile * 100)
                    
                    self.file_results[str(file_path)] = {
                        'score': float(mean_score),
                        'ci_lower': float(score_ci_lower),
                        'ci_upper': float(score_ci_upper),
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
            
            return mean_distance, ci_lower, ci_upper
        else:
            print(f"DEBUG: Skipping bootstrap (n_bootstrap=0)")
            # Original non-bootstrapped computation
            if isinstance(dataset1, DataDistribution):
                dist1 = dataset1.get_distribution(self.key)
            else:
                dist1 = self.get_distribution(dataset1)
            
            # Store per-file results
            if isinstance(dataset2, Dataset):
                files = dataset2.get_files()
            elif isinstance(dataset2, DataDistribution):
                files = dataset2.get_dataset().get_files()
            else:
                files = []
            
            for i, file_path in enumerate(files):
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
            
            if self.dimension == BenchmarkDimension.ONE_DIMENSIONAL:
                return wasserstein_distance(dist1, dist2)
            else:  # N_DIMENSIONAL
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
                ref_mu, ref_sigma = reference_dist
                # For (mu, sigma) tuples, compute distance to the mean
                if isinstance(file_dist, tuple):
                    file_mu, file_sigma = file_dist
                    # Ensure dimensions match by taking mean if needed
                    if len(file_mu.shape) > 1:
                        file_mu = np.mean(file_mu, axis=0)
                    if len(file_sigma.shape) > 2:
                        file_sigma = np.mean(file_sigma, axis=0)
                    return frechet_distance(file_mu, ref_mu, file_sigma, ref_sigma)
                else:
                    # If file_dist is a direct array, treat it as a single point
                    if len(file_dist.shape) == 1:
                        file_dist = file_dist.reshape(1, -1)
                    file_mu = np.mean(file_dist, axis=0)
                    file_sigma = np.cov(file_dist, rowvar=False) if file_dist.shape[0] > 1 else np.zeros_like(ref_sigma)
                    return frechet_distance(file_mu, ref_mu, file_sigma, ref_sigma)
            else:
                # For direct arrays, use standard distance
                if len(file_dist.shape) == 1:
                    file_dist = file_dist.reshape(1, -1)
                if len(reference_dist.shape) == 1:
                    reference_dist = reference_dist.reshape(1, -1)
                # Compute mean and covariance for both
                file_mu = np.mean(file_dist, axis=0)
                ref_mu = np.mean(reference_dist, axis=0)
                file_sigma = np.cov(file_dist, rowvar=False) if file_dist.shape[0] > 1 else np.zeros_like(np.cov(reference_dist, rowvar=False))
                ref_sigma = np.cov(reference_dist, rowvar=False)
                return frechet_distance(file_mu, ref_mu, file_sigma, ref_sigma)

    def compute_score(
        self,
        dataset: Dataset,
        reference_datasets: List[DataDistribution],
        noise_datasets: List[DataDistribution],
        n_bootstrap: int = 0,
        confidence: float = 0.95
    ) -> Tuple[float, Tuple[float, float], Tuple[str, str], Dict]:
        """
        Compute the score of the benchmark on a dataset.
        """
        print(f"\nDEBUG: compute_score called with n_bootstrap={n_bootstrap}, confidence={confidence}")
        
        # Get the best reference dataset
        reference_scores = []
        for reference_ds in reference_datasets:
            print(f"DEBUG: Computing distance for reference dataset {reference_ds.name}")
            score = self.compute_distance(
                reference_ds,
                dataset,
                n_bootstrap=n_bootstrap,
                confidence=confidence
            )
            if isinstance(score, tuple):
                print(f"DEBUG: Got bootstrap result for reference: mean={score[0]:.4f}, CI=[{score[1]:.4f}, {score[2]:.4f}]")
                reference_scores.append((score[0], score[1], score[2], reference_ds))
            else:
                print(f"DEBUG: Got non-bootstrap result for reference: {score:.4f}")
                reference_scores.append((score, None, None, reference_ds))
        reference_scores = sorted(reference_scores, key=lambda x: x[0])
        best_reference_score = reference_scores[0]
        best_reference = best_reference_score[3]
        print(f"DEBUG: Best reference dataset: {best_reference.name}")
        
        # Get the worst noise dataset
        noise_scores = []
        for noise_ds in noise_datasets:
            print(f"DEBUG: Computing distance for noise dataset {noise_ds.name}")
            score = self.compute_distance(
                noise_ds,
                dataset,
                n_bootstrap=0  # No bootstrapping needed for noise datasets
            )
            if isinstance(score, tuple):
                noise_scores.append((score[0], noise_ds))
            else:
                noise_scores.append((score, noise_ds))
        noise_scores = sorted(noise_scores, key=lambda x: x[0])
        worst_noise = noise_scores[-1]
        print(f"DEBUG: Worst noise dataset: {worst_noise[1].name}")
        
        # Compute final score
        if n_bootstrap > 0 and isinstance(best_reference_score, tuple) and len(best_reference_score) >= 3:
            print(f"DEBUG: Computing bootstrapped final score")
            # If bootstrapping is enabled, use mean and CI from best reference
            reference_mean = best_reference_score[0]
            reference_ci_lower = best_reference_score[1]
            reference_ci_upper = best_reference_score[2]
            noise_mean = worst_noise[0]
            
            # Compute normalized score with confidence interval
            score = 1 - (reference_mean / noise_mean)
            
            # Compute normalized confidence intervals
            ci_lower_norm = 1 - (reference_ci_upper / noise_mean)  # Note: upper/lower swap due to 1-x transformation
            ci_upper_norm = 1 - (reference_ci_lower / noise_mean)
            ci_width = ci_upper_norm - ci_lower_norm
            
            ci = (ci_width, confidence)
            print(f"DEBUG: Final bootstrapped score: {score:.4f}, CI width: {ci_width:.4f}")
        else:
            print(f"DEBUG: Computing non-bootstrapped final score")
            # Without bootstrapping, compute score as before
            score = 1 - (best_reference_score[0] / worst_noise[0])
            ci = (0.0, 0.0)  # No confidence interval
            print(f"DEBUG: Final non-bootstrapped score: {score:.4f}")
        
        return score, ci, (worst_noise[1].name, best_reference.name), self.file_results
