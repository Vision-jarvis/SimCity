"""
Polarization model for tracking opinion fragmentation and echo chamber dynamics.
Based on the Deffuant bounded-confidence model extended with group dynamics.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PolarizationMetrics:
    """Quantified polarization state."""
    polarization_index: float  # 0 = consensus, 1 = maximum polarization
    n_clusters: int  # Number of opinion clusters
    cluster_sizes: List[int]
    cluster_means: List[float]
    bimodality_coefficient: float
    echo_chamber_strength: float


class PolarizationModel:
    """
    Quantifies and forecasts opinion polarization.

    Uses multiple metrics:
    1. Esteban-Ray polarization index
    2. Bimodality coefficient
    3. Echo chamber strength (within-group vs between-group communication)
    4. Cluster analysis of opinion distribution

    Can be coupled with the Deffuant model for forward simulation.
    """

    def __init__(self, n_clusters_max: int = 5, er_alpha: float = 1.6):
        self.n_clusters_max = n_clusters_max
        self.er_alpha = er_alpha  # Esteban-Ray alpha parameter

    def measure(self, opinions: np.ndarray) -> PolarizationMetrics:
        """
        Measure polarization from a distribution of opinions in [0, 1].

        Args:
            opinions: Array of opinion values in [0, 1].

        Returns:
            PolarizationMetrics with all computed indices.
        """
        opinions = np.clip(opinions, 0, 1)
        n = len(opinions)

        if n < 2:
            return PolarizationMetrics(
                polarization_index=0.0, n_clusters=1,
                cluster_sizes=[n], cluster_means=[float(opinions.mean()) if n else 0.5],
                bimodality_coefficient=0.0, echo_chamber_strength=0.0,
            )

        # 1. Find opinion clusters using simple histogram-based approach
        clusters, means, sizes = self._find_clusters(opinions)

        # 2. Esteban-Ray polarization index
        er_index = self._esteban_ray(opinions, clusters, means, sizes)

        # 3. Bimodality coefficient
        bimod = self._bimodality_coefficient(opinions)

        # 4. Echo chamber strength
        echo = self._echo_chamber_strength(opinions, clusters)

        # Composite polarization index
        pol_index = 0.4 * er_index + 0.3 * bimod + 0.3 * echo

        return PolarizationMetrics(
            polarization_index=round(pol_index, 4),
            n_clusters=len(means),
            cluster_sizes=sizes,
            cluster_means=[round(m, 4) for m in means],
            bimodality_coefficient=round(bimod, 4),
            echo_chamber_strength=round(echo, 4),
        )

    def _find_clusters(
        self, opinions: np.ndarray
    ) -> Tuple[np.ndarray, List[float], List[int]]:
        """
        Simple k-means-like clustering of opinions.
        Automatically determines number of clusters via gap statistic.
        """
        # Try different k values, pick best
        best_k, best_score = 1, -1
        best_labels = np.zeros(len(opinions), dtype=int)
        best_means = [float(opinions.mean())]

        for k in range(2, self.n_clusters_max + 1):
            if k > len(opinions):
                break

            # Simple uniform initialization
            centers = np.linspace(opinions.min(), opinions.max(), k)

            for _ in range(20):  # Max iterations
                # Assign to nearest center
                dists = np.abs(opinions[:, None] - centers[None, :])
                labels = dists.argmin(axis=1)

                # Update centers
                new_centers = np.array([
                    opinions[labels == i].mean() if (labels == i).any() else centers[i]
                    for i in range(k)
                ])

                if np.allclose(centers, new_centers, atol=1e-6):
                    break
                centers = new_centers

            # Silhouette-like score
            intra = sum(
                np.abs(opinions[labels == i] - centers[i]).mean()
                for i in range(k) if (labels == i).any()
            ) / k

            if k > 1:
                inter = np.min([
                    np.abs(centers[i] - centers[j])
                    for i in range(k) for j in range(i + 1, k)
                ])
            else:
                inter = 0

            score = (inter - intra) / max(inter, intra, 1e-8) if k > 1 else 0

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_means = centers.tolist()

        # Remove empty clusters
        active = [i for i in range(best_k) if (best_labels == i).any()]
        means = [best_means[i] for i in active]
        sizes = [int((best_labels == i).sum()) for i in active]

        return best_labels, means, sizes

    def _esteban_ray(
        self, opinions: np.ndarray, labels: np.ndarray,
        means: List[float], sizes: List[int]
    ) -> float:
        """Compute Esteban-Ray polarization index."""
        n = len(opinions)
        if n == 0 or len(means) < 2:
            return 0.0

        # ER = sum_i sum_j p_i^(1+alpha) * p_j * |y_i - y_j|
        fracs = np.array(sizes) / n
        er = 0.0
        for i in range(len(means)):
            for j in range(len(means)):
                er += fracs[i] ** (1 + self.er_alpha) * fracs[j] * abs(means[i] - means[j])

        # Normalize to [0, 1]
        return min(1.0, er / 0.5)

    def _bimodality_coefficient(self, opinions: np.ndarray) -> float:
        """
        Sarle's bimodality coefficient.
        BC > 0.555 suggests bimodal distribution.
        """
        from scipy.stats import skew, kurtosis
        n = len(opinions)
        if n < 4:
            return 0.0

        try:
            s = skew(opinions)
            k = kurtosis(opinions, fisher=True)
            bc = (s ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
            return min(1.0, max(0.0, bc))
        except Exception:
            # Fallback: simple variance-based measure
            return min(1.0, opinions.std() * 4)

    def _echo_chamber_strength(
        self, opinions: np.ndarray, labels: np.ndarray
    ) -> float:
        """
        Measure echo chamber strength as ratio of within-group variance
        to between-group variance.
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        # Within-group variance
        within = np.mean([
            opinions[labels == l].var() if (labels == l).sum() > 1 else 0
            for l in unique_labels
        ])

        # Between-group variance
        group_means = np.array([opinions[labels == l].mean() for l in unique_labels])
        between = group_means.var()

        if between < 1e-8:
            return 0.0

        # High ratio = strong echo chambers (low within, high between)
        return min(1.0, between / (within + between + 1e-8))

    def forecast_polarization(
        self,
        current_opinions: np.ndarray,
        steps: int = 10,
        mu: float = 0.5,
        epsilon: float = 0.3,
    ) -> List[PolarizationMetrics]:
        """
        Forecast polarization trajectory using Deffuant dynamics.

        Args:
            current_opinions: Current opinion distribution.
            steps: Number of simulation steps.
            mu: Convergence parameter (how fast opinions change).
            epsilon: Bounded confidence threshold.

        Returns:
            List of PolarizationMetrics, one per step.
        """
        opinions = current_opinions.copy()
        trajectory = [self.measure(opinions)]

        for _ in range(steps):
            # Random pairwise interactions
            n = len(opinions)
            for _ in range(n):
                i, j = np.random.randint(0, n, size=2)
                if abs(opinions[i] - opinions[j]) < epsilon:
                    delta = mu * (opinions[j] - opinions[i])
                    opinions[i] += delta
                    opinions[j] -= delta

            opinions = np.clip(opinions, 0, 1)
            trajectory.append(self.measure(opinions))

        return trajectory
