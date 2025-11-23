# train per tile PCA
# test_train.py
import argparse
import json
import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from test_model import (
    TRAIN_TIMES,
    load_lightmap_stack,
    train_pca_for_lightmap,
    compute_psnr,
    PCALightmapModel,
    PCAClusterModel,
    segment_mask,
    time_str_to_token,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCA-based lightmap compression baseline.")
    parser.add_argument("--dataset", type=str, default="../Data/Data_HPRC", help="Dataset root directory")
    parser.add_argument("--param_dir", type=str, default="./test_parameters", help="Directory to store PCA models")
    parser.add_argument("--lightmap_limit", type=int, default=5, help="Max lightmaps to process (None for all)")
    parser.add_argument("--components", type=int, default=12, help="Number of PCA components for deployment")
    parser.add_argument(
        "--eval_components",
        type=int,
        nargs="*",
        default=[6, 8, 10, 12],
        help="Component counts to evaluate for reporting (includes --components if not listed)",
    )
    parser.add_argument("--plot", action="store_true", help="Plot random pixel curve for sanity check")
    parser.add_argument("--erosion_iterations", type=int, default=2, help="Mask erosion iterations for clustering")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size used during erosion")
    return parser.parse_args()


def ensure_eval_list(eval_components: List[int], chosen: int) -> List[int]:
    if chosen not in eval_components:
        return eval_components + [chosen]
    return eval_components


def maybe_plot_pixel(stack: np.ndarray, component_counts: List[int]):
    samples = stack.reshape(stack.shape[0], -1)
    idx = random.randrange(samples.shape[0])
    original = samples[idx]

    plt.figure(figsize=(12, 4))
    plt.plot(original[1::3], label="Original (G channel)", linewidth=2)
    for nc in component_counts:
        pca, coeffs = train_pca_for_lightmap(stack, nc)
        rec = pca.inverse_transform(coeffs)[idx]
        plt.plot(rec[1::3], "--", label=f"PCA {nc} comp", linewidth=2)

    plt.title("Random Pixel – Green Channel over 26 time steps (linear space)")
    plt.xlabel("Time index (0–25)")
    plt.ylabel("Light intensity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def fit_cluster(stack: np.ndarray, n_components: int, eps: float = 1e-6):
    flat = stack.reshape(stack.shape[0], -1)
    var = np.var(flat, axis=0).sum()
    if var < eps:
        mean = flat[0]
        components = np.zeros((0, flat.shape[1]), dtype=np.float32)
        coeffs = np.zeros((flat.shape[0], 0), dtype=np.float32)
        recon = np.tile(mean, (flat.shape[0], 1))
        return components, mean, coeffs, recon

    n_comp = max(1, min(n_components, stack.shape[0]))
    pca, coeffs = train_pca_for_lightmap(stack, n_comp)
    recon = pca.inverse_transform(coeffs)
    return pca.components_, pca.mean_, coeffs, recon


def main():
    args = parse_args()
    os.makedirs(args.param_dir, exist_ok=True)

    with open(os.path.join(args.dataset, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    lightmaps = config["lightmap_list"]
    if args.lightmap_limit is not None:
        lightmaps = lightmaps[: args.lightmap_limit]

    eval_components = ensure_eval_list(args.eval_components, args.components)
    time_tokens = [time_str_to_token(t) for t in TRAIN_TIMES]

    per_component_scores = {nc: [] for nc in eval_components}

    print(f"Linear-space PCA baseline on {len(lightmaps)} lightmaps\n")
    for idx, lm_cfg in enumerate(lightmaps):
        lid = f"{lm_cfg['level']}_{lm_cfg['id']}"
        H, W = lm_cfg["resolution"]["height"], lm_cfg["resolution"]["width"]
        print(f"[{idx + 1}/{len(lightmaps)}] {lid} ({H}x{W}) ... ", end="")

        stack, valid_idx, mask_bool = load_lightmap_stack(lm_cfg, args.dataset)
        H_flat = H * W
        label_map, num_labels = segment_mask(mask_bool, H, W, iterations=args.erosion_iterations, kernel_size=args.kernel_size)
        valid_labels = label_map.reshape(-1)[valid_idx]
        unique_labels = np.unique(valid_labels)
        unique_labels = unique_labels[unique_labels > 0]
        if unique_labels.size == 0:
            unique_labels = np.array([1], dtype=np.int32)
            valid_labels = np.ones_like(valid_labels, dtype=np.int32)
        data_flat = stack.reshape(stack.shape[0], -1)
        print(f"{stack.shape[0] // 1000}k valid pixels", end=" -> ")

        chosen_clusters = None
        for nc in eval_components:
            recon = np.zeros_like(data_flat)
            cluster_storage = []
            for label in unique_labels:
                cluster_mask = valid_labels == label
                cluster_stack = stack[cluster_mask]
                if cluster_stack.size == 0:
                    continue
                components, mean, coeffs, cluster_recon = fit_cluster(cluster_stack, nc)
                recon[cluster_mask] = cluster_recon
                cluster_storage.append((label, cluster_mask, components, mean, coeffs))
            psnr = compute_psnr(data_flat, recon)
            per_component_scores[nc].append(psnr)
            if nc == args.components:
                chosen_clusters = cluster_storage
            print(f"{nc}c:{psnr:.2f}dB ", end="")

        print()

        if chosen_clusters is None:
            fallback_storage = []
            for label in unique_labels:
                cluster_mask = valid_labels == label
                cluster_stack = stack[cluster_mask]
                if cluster_stack.size == 0:
                    continue
                components, mean, coeffs, _ = fit_cluster(cluster_stack, args.components)
                fallback_storage.append((label, cluster_mask, components, mean, coeffs))
            chosen_clusters = fallback_storage

        cluster_models: List[PCAClusterModel] = []
        cluster_labels: List[int] = []
        for label, cluster_mask, components, mean, coeffs in chosen_clusters:
            pixel_indices = valid_idx[cluster_mask]
            cluster_models.append(
                PCAClusterModel(
                    components=components,
                    mean=mean,
                    coeffs=coeffs,
                    pixel_indices=pixel_indices,
                    component_count=components.shape[0],
                )
            )
            cluster_labels.append(int(label))

        model = PCALightmapModel(
            resolution=(H, W),
            time_tokens=time_tokens,
            cluster_labels=cluster_labels,
            clusters=cluster_models,
        )
        save_path = os.path.join(args.param_dir, f"pca_{lid}.npz")
        model.to_npz(save_path)
        print(f"   Saved parameters -> {save_path}")

        if args.plot and idx == 0:
            maybe_plot_pixel(stack, component_counts=sorted(set(eval_components)))

    print("\n" + "=" * 60)
    print("Linear-space PCA upper bound (per-component PSNR)")
    print("=" * 60)
    print(f"{'Components':<12}{'Average':<12}{'Best':<12}{'Worst':<12}")
    for nc in sorted(per_component_scores.keys()):
        scores = np.array(per_component_scores[nc])
        if scores.size == 0:
            continue
        print(f"{nc:<12}{scores.mean():<12.2f}{scores.max():<12.2f}{scores.min():<12.2f}")


if __name__ == "__main__":
    main()
