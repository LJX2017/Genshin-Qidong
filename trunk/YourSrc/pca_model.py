import numpy as np


class PCALightmapModel:
    """
    PCA-based lightmap compressor.

    - Trains PCA on 4x4x3xT blocks.
    - Keeps top K principal components.
    - Stores:
        * mean vector (float32)
        * basis matrix (float32)
        * per-component min/scale for int8 quantization (float32)
        * int8 coefficients
        * header with H, W, block_size, K, T, grid_h, grid_w

    Binary layout (little-endian):
    [ int32 x 7 ]: H, W, block_size, K, T, grid_h, grid_w
    [ float32 x D ]: mean vector
    [ float32 x (D*K) ]: basis matrix (column-major: D rows, K cols)
    [ float32 x K ]: min_val for scores
    [ float32 x K ]: scale for scores  (scale = max - min, with 0 replaced by 1)
    [ uint8   x (grid_h * grid_w * K) ]: quantized scores (row-major over blocks)
    """

    def __init__(self, block_size: int = 4, K: int = 100):
        self.block_size = block_size
        self.K = K

        # Geometry
        self.H = None
        self.W = None
        self.T = None
        self.grid_h = None
        self.grid_w = None

        # PCA parameters
        self.mean = None         # [D]
        self.basis = None        # [D, K]
        self.min_val = None      # [K]
        self.scale = None        # [K]
        self.coeff_q = None      # [N_blocks, K], uint8

    # ------------------------------------------------------------------
    # Training / fitting
    # ------------------------------------------------------------------
    def fit(self, blocks_np: np.ndarray, H: int, W: int, T: int):
        """
        Fit PCA model on given blocks.

        Args:
            blocks_np: [N_blocks, D] float32 array, where
                        D = block_size * block_size * 3 * T
            H, W: full lightmap resolution
            T:    number of time steps
        """
        assert blocks_np.ndim == 2, "blocks_np must be [N_blocks, D]"
        N, D = blocks_np.shape

        self.H = int(H)
        self.W = int(W)
        self.T = int(T)

        B = self.block_size
        self.grid_h = self.H // B
        self.grid_w = self.W // B

        # Safety check
        expected_D = B * B * 3 * self.T
        if D != expected_D:
            raise ValueError(f"Expected D={expected_D}, got D={D}")

        # Compute mean
        mean_vec = blocks_np.mean(axis=0)
        Xc = blocks_np - mean_vec[None, :]

        # Covariance and eigen decomposition
        # cov: [D, D]
        cov = (Xc.T @ Xc) / (N - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvectors by descending eigenvalues
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        # Keep top K
        K = self.K
        basis = eigvecs[:, :K]  # [D, K]

        # Project to scores
        scores = Xc @ basis  # [N, K]

        # Per-component min and scale for int8 quantization
        min_val = scores.min(axis=0)
        max_val = scores.max(axis=0)
        scale = max_val - min_val
        scale[scale == 0] = 1.0

        # Normalize to [0, 1], then to [0, 255] and quantize
        norm = (scores - min_val[None, :]) / scale[None, :]
        q = np.round(norm * 255.0)
        q = np.clip(q, 0, 255).astype(np.uint8)

        # Store parameters (all float32 except coeff_q)
        self.mean = mean_vec.astype(np.float32)
        self.basis = basis.astype(np.float32)
        self.min_val = min_val.astype(np.float32)
        self.scale = scale.astype(np.float32)
        self.coeff_q = q  # uint8

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str):
        """
        Save model to a single raw binary file.
        Layout described in the class docstring.
        """
        if any(x is None for x in [self.mean, self.basis, self.min_val,
                                   self.scale, self.coeff_q]):
            raise RuntimeError("Model parameters are not initialized. Call fit() first.")

        header = np.array([
            self.H,
            self.W,
            self.block_size,
            self.K,
            self.T,
            self.grid_h,
            self.grid_w
        ], dtype=np.int32)

        with open(path, "wb") as f:
            header.tofile(f)
            self.mean.astype(np.float32).tofile(f)
            self.basis.astype(np.float32).tofile(f)
            self.min_val.astype(np.float32).tofile(f)
            self.scale.astype(np.float32).tofile(f)
            self.coeff_q.tofile(f)

    @classmethod
    def load(cls, path: str) -> "PCALightmapModel":
        """
        Load model from the raw binary file saved by save().
        """
        with open(path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=7)
            if header.size != 7:
                raise RuntimeError("Failed to read PCA header from file")

            H, W, block_size, K, T, grid_h, grid_w = header.tolist()

            model = cls(block_size=block_size, K=K)
            model.H = int(H)
            model.W = int(W)
            model.T = int(T)
            model.grid_h = int(grid_h)
            model.grid_w = int(grid_w)

            B = model.block_size
            D = B * B * 3 * model.T
            N_blocks = model.grid_h * model.grid_w

            # Read arrays
            mean = np.fromfile(f, dtype=np.float32, count=D)
            basis = np.fromfile(f, dtype=np.float32, count=D * K)
            min_val = np.fromfile(f, dtype=np.float32, count=K)
            scale = np.fromfile(f, dtype=np.float32, count=K)
            coeff_q = np.fromfile(f, dtype=np.uint8, count=N_blocks * K)

            if mean.size != D or basis.size != D * K or \
               min_val.size != K or scale.size != K or \
               coeff_q.size != N_blocks * K:
                raise RuntimeError("File size does not match expected PCA layout")

            model.mean = mean
            model.basis = basis.reshape(D, K)
            model.min_val = min_val
            model.scale = scale
            model.coeff_q = coeff_q.reshape(N_blocks, K)

        return model

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode_blocks(self) -> np.ndarray:
        """
        Decode all blocks back into float32.

        Returns:
            blocks: [N_blocks, block_size, block_size, 3, T]
        """
        if self.coeff_q is None:
            raise RuntimeError("Model has no coefficients loaded")

        B = self.block_size
        K = self.K
        T = self.T
        N_blocks = self.grid_h * self.grid_w
        D = B * B * 3 * T

        # Dequantize scores
        q = self.coeff_q.astype(np.float32)          # [N_blocks, K]
        norm = q / 255.0
        scores = norm * self.scale[None, :] + self.min_val[None, :]  # [N_blocks, K]

        # Reconstruct blocks in D-dim space
        X = scores @ self.basis.T + self.mean[None, :]  # [N_blocks, D]

        # Reshape into spatial-temporal blocks
        blocks = X.reshape(N_blocks, B, B, 3, T)  # [N_blocks, B, B, 3, T]
        return blocks
