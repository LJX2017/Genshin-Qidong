import argparse
import json
import os
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from PCA_VQ_Model import PCA_VQ_Model

TRAIN_TIMES = [
    "0", "100", "200", "300", "400", "500", "590", "600", "700", "800", "900", "1000",
    "1100", "1200", "1300", "1400", "1500", "1600", "1700", "1800", "1810", "1900",
    "2000", "2100", "2200", "2300"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../Data/Data_HPRC")
    parser.add_argument("--output_dir", type=str, default="./pca_vq_parameters")
    parser.add_argument("--lightmap_count", type=int, default=205)

    # Settings
    parser.add_argument("--n_components", type=int, default=12)
    parser.add_argument("--n_clusters", type=int, default=4096)
    parser.add_argument("--kmeans_batch", type=int, default=4096 * 2)
    return parser.parse_args()


def prepare_train_data(lightmap, dataset_root):
    mask_path = os.path.join(dataset_root, "Data", lightmap["masks"]["0"])
    mask_bool = np.fromfile(mask_path, dtype=np.int8).reshape(-1) > 0
    valid_indices = np.where(mask_bool)[0]

    pixel_history = []
    for t_str in TRAIN_TIMES:
        path = os.path.join(dataset_root, "Data", lightmap["lightmaps"][t_str])
        data_lm = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
        pixel_history.append(data_lm[valid_indices])

    return np.stack(pixel_history, axis=1).reshape(len(valid_indices), -1), valid_indices


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.dataset, "config.json"), "r") as f:
        config = json.load(f)

    lightmaps = config["lightmap_list"][:args.lightmap_count]
    print(f"Training {len(lightmaps)} maps. Saving to {args.output_dir}...")

    for i, lm in enumerate(lightmaps):
        lid = f"{lm['level']}_{lm['id']}"
        H, W = lm["resolution"]["height"], lm["resolution"]["width"]
        print(f"[{i + 1}/{len(lightmaps)}] {lid}...", end=" ", flush=True)

        # 1. Process
        train_data, valid_indices = prepare_train_data(lm, args.dataset)

        pca = PCA(n_components=args.n_components)
        reduced_data = pca.fit_transform(train_data)
        basis = pca.components_.astype(np.float32)
        mean = pca.mean_.astype(np.float32)

        kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, batch_size=args.kmeans_batch, n_init=3)
        labels = kmeans.fit_predict(reduced_data)
        centroids = kmeans.cluster_centers_.astype(np.float32)

        # 2. Reconstruct Index Map
        full_index_map = np.zeros(H * W, dtype=np.uint32)
        full_index_map[valid_indices] = labels

        # 3. Save RAW
        save_path = os.path.join(args.output_dir, f"vq_{lid}.bin")
        PCA_VQ_Model.save_raw(
            save_path,
            args.n_clusters, H, W,
            centroids, full_index_map, basis, mean
        )
        print("Done.")


if __name__ == "__main__":
    main()