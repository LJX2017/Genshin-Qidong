import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Sequence

import cv2
import numpy as np
from sklearn.decomposition import PCA

# Canonical training/evaluation time steps (scaled by 100 for file indexing)
TRAIN_TIMES = [
    "0", "100", "200", "300", "400", "500", "600", "700", "800", "900", "1000",
    "1100", "1200", "1300", "1400", "1500", "1600", "1700", "1800", "1900",
    "2000", "2100", "2200", "2300", "590", "1810",
]


def time_str_to_token(t_str: str) -> int:
    """Convert dataset time string to integer token (e.g., '590' -> 590)."""
    return int(t_str)


def encode_time_value(value: float) -> int:
    """
    Convert a floating-point time (e.g., 5.9, 18.0) to dataset token representation.
    Integer hours (e.g., 5) become 500, fractional hours become value * 100 rounded.
    """
    scaled = int(round(float(value) * 100))
    return scaled


def load_lightmap_stack(lightmap_cfg: dict, dataset_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replicates dataset loading used in the PCA experiment:
    - Reads the union mask once (time '0') and keeps only valid pixels.
    - Stacks RGB samples for the canonical time list.
    Returns:
        stack: (num_valid_pixels, len(TRAIN_TIMES), 3)
        valid_indices: flat indices (wrt H*W ordering) indicating valid pixels
        mask_bool: boolean mask of shape (H*W) for valid pixels
    """
    mask_path = os.path.join(dataset_root, "Data", lightmap_cfg["masks"]["0"])
    mask_bool = np.fromfile(mask_path, dtype=np.int8).reshape(-1) > 0
    valid_indices = np.where(mask_bool)[0]

    pixel_history: List[np.ndarray] = []
    for t_str in TRAIN_TIMES:
        path = os.path.join(dataset_root, "Data", lightmap_cfg["lightmaps"][t_str])
        data = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
        pixel_history.append(data[valid_indices])

    stack = np.stack(pixel_history, axis=1)  # (N_valid, len(times), 3)
    return stack, valid_indices, mask_bool


def segment_mask(mask_bool: np.ndarray, height: int, width: int, iterations: int = 1, kernel_size: int = 3):
    """
    Split the mask into spatially separated islands using morphological erosion
    followed by connected components and region growing.
    Returns label_map shaped (H, W) with 0 as background.
    """
    mask_uint8 = (mask_bool.reshape(height, width) * 255).astype(np.uint8)
    if np.sum(mask_uint8) == 0:
        raise ValueError("Mask is empty; cannot segment.")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=max(1, iterations))

    num_labels, core_labels = cv2.connectedComponents(eroded, connectivity=4)
    if num_labels <= 1:
        # Fallback: single cluster identical to original mask
        base_labels = np.zeros_like(core_labels, dtype=np.int32)
        base_labels[mask_uint8 > 0] = 1
        return base_labels, 1

    # Region growing to cover entire mask
    label_map = core_labels.copy()
    mask_arr = mask_uint8 > 0
    from collections import deque

    queue = deque()
    ys, xs = np.nonzero(label_map > 0)
    for y, x in zip(ys, xs):
        queue.append((y, x))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        y, x = queue.popleft()
        label = label_map[y, x]
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if mask_arr[ny, nx] and label_map[ny, nx] == 0:
                    label_map[ny, nx] = label
                    queue.append((ny, nx))

    # Any remaining masked pixel without label (e.g., isolated pixel removed by erosion)
    unlabeled = (label_map == 0) & mask_arr
    if np.any(unlabeled):
        # Assign new labels to these isolated pixels
        next_label = label_map.max() + 1
        label_map[unlabeled] = next_label
        num_labels = next_label
    else:
        num_labels = label_map.max()

    return label_map, num_labels


@dataclass
class PCAClusterModel:
    components: np.ndarray  # (n_components, len(times) * 3)
    mean: np.ndarray        # (len(times) * 3,)
    coeffs: np.ndarray      # (n_pixels_in_cluster, n_components)
    pixel_indices: np.ndarray  # flattened pixel indices
    component_count: int


@dataclass
class PCALightmapModel:
    """Holds PCA decomposition info for multiple spatial clusters."""

    resolution: Tuple[int, int]
    time_tokens: List[int]
    cluster_labels: List[int]
    clusters: List[PCAClusterModel]

    @property
    def time_count(self) -> int:
        return len(self.time_tokens)

    def _object_array(self, data: Sequence[np.ndarray]):
        return np.array([arr for arr in data], dtype=object)

    def to_npz(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        comp_list = [cluster.components.astype(np.float32) for cluster in self.clusters]
        mean_list = [cluster.mean.astype(np.float32) for cluster in self.clusters]
        coeff_list = [cluster.coeffs.astype(np.float32) for cluster in self.clusters]
        idx_list = [cluster.pixel_indices.astype(np.int64) for cluster in self.clusters]
        comp_counts = [cluster.component_count for cluster in self.clusters]

        np.savez_compressed(
            path,
            resolution=np.array(self.resolution, dtype=np.int32),
            time_tokens=np.array(self.time_tokens, dtype=np.int32),
            cluster_labels=np.array(self.cluster_labels, dtype=np.int32),
            cluster_components=self._object_array(comp_list),
            cluster_means=self._object_array(mean_list),
            cluster_coeffs=self._object_array(coeff_list),
            cluster_indices=self._object_array(idx_list),
            cluster_component_counts=np.array(comp_counts, dtype=np.int32),
        )

    @staticmethod
    def from_npz(path: str) -> "PCALightmapModel":
        data = np.load(path, allow_pickle=True)
        resolution = tuple(int(x) for x in data["resolution"])
        time_tokens = data["time_tokens"].astype(np.int32).tolist()
        cluster_labels = data["cluster_labels"].astype(np.int32).tolist()
        comp_counts = data["cluster_component_counts"].astype(np.int32).tolist()
        components = data["cluster_components"]
        means = data["cluster_means"]
        coeffs = data["cluster_coeffs"]
        indices = data["cluster_indices"]

        clusters: List[PCAClusterModel] = []
        for comp_arr, mean_arr, coeff_arr, idx_arr, comp_count in zip(
            components, means, coeffs, indices, comp_counts
        ):
            clusters.append(
                PCAClusterModel(
                    components=np.asarray(comp_arr, dtype=np.float32),
                    mean=np.asarray(mean_arr, dtype=np.float32),
                    coeffs=np.asarray(coeff_arr, dtype=np.float32),
                    pixel_indices=np.asarray(idx_arr, dtype=np.int64),
                    component_count=int(comp_count),
                )
            )

        return PCALightmapModel(
            resolution=resolution,
            time_tokens=time_tokens,
            cluster_labels=cluster_labels,
            clusters=clusters,
        )


def train_pca_for_lightmap(stack: np.ndarray, n_components: int) -> Tuple[PCA, np.ndarray]:
    """
    Fit PCA on flattened sequences: stack shape (N, T, 3) -> (N, T*3).
    Returns fitted PCA object and the transformed coefficients.
    """
    samples = stack.reshape(stack.shape[0], -1)
    pca = PCA(n_components=n_components, random_state=42)
    coeffs = pca.fit_transform(samples)
    return pca, coeffs


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((original - reconstructed) ** 2)
    if mse <= 0:
        return 99.99
    peak = max(original.max(), 1.0)
    return 20 * np.log10(peak / np.sqrt(mse))


def prepare_time_lookup(time_tokens: List[int]) -> Dict[int, int]:
    return {token: idx for idx, token in enumerate(time_tokens)}
