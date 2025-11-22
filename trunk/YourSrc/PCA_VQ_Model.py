import torch
import numpy as np
import os


class PCA_VQ_Model:
    def __init__(self, centroids, indices, basis, mean, h, w):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load Data to GPU
        c_tensor = torch.from_numpy(centroids).to(self.device, dtype=torch.float32)
        b_tensor = torch.from_numpy(basis).to(self.device, dtype=torch.float32)
        m_tensor = torch.from_numpy(mean).to(self.device, dtype=torch.float32)
        self.indices = torch.from_numpy(indices.astype(np.int64)).to(self.device)
        self.h = int(h)
        self.w = int(w)

        # 2. Pre-compute (Fuse PCA+VQ)
        # Result: [K, 72] -> Reshaped to [K, T, 3]
        self.decoded_codebook = (c_tensor @ b_tensor) + m_tensor
        T = self.decoded_codebook.shape[1] // 3
        self.decoded_codebook = self.decoded_codebook.view(c_tensor.shape[0], T, 3)

    @staticmethod
    def save_raw(path, n_clusters, h, w, centroids, indices, basis, mean):
        """
        Mimics ExampleTrain.py: Flattens arrays and dumps raw binary.
        Layout: [Header (4 ints)] + [All Floats] + [All Indices]
        """
        # 1. Prepare Header
        # We need H, W, K, and N_Comps to know how to read the file back
        n_components = basis.shape[0]
        header = np.array([h, w, n_clusters, n_components], dtype=np.int32)

        # 2. Prepare Floats (Concatenate like the example code)
        # We flatten Centroids, Basis, and Mean into one long float list
        float_data = np.concatenate([
            centroids.flatten(),
            basis.flatten(),
            mean.flatten()
        ]).astype(np.float32)

        # 3. Prepare Indices
        # Determine the smallest safe integer type to save space
        if n_clusters < 256:
            idx_dtype = np.uint8
        elif n_clusters < 65536:
            idx_dtype = np.uint16
        else:
            idx_dtype = np.uint32
        int_data = indices.astype(idx_dtype)

        # 4. Write to File (Sequential raw dump)
        with open(path, "wb") as f:
            header.tofile(f)  # 16 bytes
            float_data.tofile(f)  # Bulk floats
            int_data.tofile(f)  # Bulk ints

    @staticmethod
    def load_raw(path, device):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

        with open(path, "rb") as f:
            # 1. Read Header (4 ints = 16 bytes)
            # h, w, n_clusters, n_components
            header = np.fromfile(f, dtype=np.int32, count=4)
            h, w, n_clusters, n_components = header

            # 2. Calculate Sizes
            # We derive Time steps (T) from the mean vector later
            # Centroids: K * n_comp
            # Basis: n_comp * (T*3)
            # Mean: (T*3)

            # Since we don't know T yet, we read Centroids first, then Basis+Mean?
            # Actually, to keep it simple, we read the exact float count we wrote.

            # Wait, we need to know T to separate Basis and Mean.
            # Let's calculate file size to deduce T.
            file_size = os.path.getsize(path)
            header_size = 16

            # Index size
            if n_clusters < 256:
                idx_bytes = 1; idx_dtype = np.uint8
            elif n_clusters < 65536:
                idx_bytes = 2; idx_dtype = np.uint16
            else:
                idx_bytes = 4; idx_dtype = np.uint32

            indices_size = h * w * idx_bytes

            # Remaining bytes are floats
            float_bytes = file_size - header_size - indices_size
            total_floats = float_bytes // 4

            # Solve for Dims (Total dimensions per pixel)
            # Total Floats = (K * Comp) + (Comp * Dims) + (Dims)
            # Total Floats = K*Comp + Dims * (Comp + 1)
            # Dims = (Total_Floats - K*Comp) / (Comp + 1)

            val_1 = n_clusters * n_components
            dims = (total_floats - val_1) // (n_components + 1)
            dims = int(dims)

            # 3. Read All Floats
            all_floats = np.fromfile(f, dtype=np.float32, count=total_floats)

            # Slicing
            ptr = 0

            # Centroids
            end = ptr + (n_clusters * n_components)
            centroids = all_floats[ptr:end].reshape(n_clusters, n_components)
            ptr = end

            # Basis
            end = ptr + (n_components * dims)
            basis = all_floats[ptr:end].reshape(n_components, dims)
            ptr = end

            # Mean
            end = ptr + dims
            mean = all_floats[ptr:end]

            # 4. Read Indices
            indices = np.fromfile(f, dtype=idx_dtype, count=h * w)

        return PCA_VQ_Model(centroids, indices, basis, mean, h, w)

    def reconstruct(self, current_time):
        # Time Mapping Logic
        t_idx = int(round(current_time))
        if abs(current_time - 5.9) < 0.1:
            t_idx = 24
        elif abs(current_time - 18.1) < 0.1:
            t_idx = 25
        t_idx = min(t_idx, self.decoded_codebook.shape[1] - 1)

        # Lookup
        colors = self.decoded_codebook[:, t_idx, :]
        flat = torch.index_select(colors, 0, self.indices)
        self.current_result = flat.view(1, self.h, self.w, 3).permute(0, 3, 1, 2).contiguous()

    def get_result(self):
        return self.current_result

    def random_test(self, coord):
        y, x, t = int(coord[0, 0]), int(coord[0, 1]), coord[0, 2].item()
        idx = y * self.w + x
        centroid_id = self.indices[idx]

        t_idx = int(round(t))
        if abs(t - 5.9) < 0.1:
            t_idx = 24
        elif abs(t - 18.1) < 0.1:
            t_idx = 25
        t_idx = min(t_idx, self.decoded_codebook.shape[1] - 1)

        val = self.decoded_codebook[centroid_id, t_idx, :]
        return val.view(1, 3)