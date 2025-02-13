"""
This module contains functions for bootstrapping distributions and computing statistics.
"""

from typing import Tuple, List, Optional, Union
import numpy as np
from .distances import wasserstein_distance, frechet_distance


def bootstrap_reference_distribution(
    reference_distribution: np.ndarray,
    n_bootstrap: int = 1000,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Generate bootstrap samples from a reference distribution.
    Works for both 1D and N-dimensional distributions.
    
    Args:
        reference_distribution: The reference distribution to bootstrap from
        n_bootstrap: Number of bootstrap samples to generate
        sample_size: Size of each bootstrap sample. If None, uses original size
        seed: Random seed for reproducibility
        
    Returns:
        List of bootstrap sample distributions
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Handle both 1D and ND distributions
    if len(reference_distribution.shape) == 1:
        if sample_size is None:
            sample_size = len(reference_distribution)
    else:
        if sample_size is None:
            sample_size = reference_distribution.shape[0]
            
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(
            len(reference_distribution) if len(reference_distribution.shape) == 1 
            else reference_distribution.shape[0],
            size=sample_size,
            replace=True
        )
        bootstrap_sample = reference_distribution[indices]
        bootstrap_samples.append(bootstrap_sample)
        
    return bootstrap_samples


def compute_bootstrap_statistics(
    synthetic_distribution: np.ndarray,
    reference_distribution: np.ndarray,
    n_bootstrap: int = 1000,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute bootstrap statistics by resampling the reference distribution and comparing
    to the fixed synthetic distribution.
    
    Args:
        synthetic_distribution: Fixed distribution from synthetic speech
        reference_distribution: Reference distribution to bootstrap from
        n_bootstrap: Number of bootstrap samples
        sample_size: Size of each bootstrap sample
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (mean_score, (ci_lower, ci_upper))
        where ci_lower and ci_upper are the 2.5th and 97.5th percentiles
    """
    # Generate bootstrap samples from reference distribution only
    bootstrap_samples = bootstrap_reference_distribution(
        reference_distribution,
        n_bootstrap=n_bootstrap,
        sample_size=sample_size,
        seed=seed
    )
    
    # Compute distances between fixed synthetic and each bootstrapped reference
    distances = []
    for bootstrap_sample in bootstrap_samples:
        if len(reference_distribution.shape) == 1:
            distance = wasserstein_distance(synthetic_distribution, bootstrap_sample)
        else:
            distance = frechet_distance(synthetic_distribution, bootstrap_sample)
        distances.append(distance)
    
    # Compute statistics
    mean_distance = np.mean(distances)
    ci_lower = np.percentile(distances, 2.5)
    ci_upper = np.percentile(distances, 97.5)
    
    return mean_distance, (ci_lower, ci_upper)


def bootstrap_benchmark_reference(
    benchmark: "Benchmark",
    reference_dataset: "Dataset",
    n_bootstrap: int = 100,
    sample_size: Optional[int] = 1000,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Generate bootstrap samples of the reference distribution for a benchmark.
    This can be used to precompute bootstrapped reference distributions.
    
    Args:
        benchmark: The benchmark to use for getting distributions
        reference_dataset: Reference dataset (e.g. AniSpeech)
        n_bootstrap: Number of bootstrap samples
        sample_size: Size of each bootstrap sample
        seed: Random seed for reproducibility
        
    Returns:
        List of bootstrapped reference distributions that can be used for evaluation
    """
    # Get the reference distribution
    reference_dist = benchmark.get_distribution(reference_dataset)
    
    # Generate bootstrap samples
    return bootstrap_reference_distribution(
        reference_dist,
        n_bootstrap=n_bootstrap,
        sample_size=sample_size,
        seed=seed
    )