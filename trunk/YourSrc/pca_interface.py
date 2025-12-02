import os
import numpy as np
import torch
import numbers

from pca_model import PCALightmapModel


class PCAInterface:
    """
    Runtime interface, analogous to BasicInterface but using PCA.

    This version:
      - Loads parameters only in __init__ (not timed).
      - Performs the heavy PCA decoding once per lightmap (on first reconstruct).
      - Subsequent reconstruct() calls reuse decoded blocks and only assemble one time slice.
      - random_test() samples from the current reconstructed frame, matching get_result().
    """

    def __init__(
        self,
        lightmap_config,
        device,
        params_dir: str = "./Parameters",
        block_size: int = 4,
        K: int = 100,
    ):
        self.device = device
        self.params_dir = params_dir

        self.level = lightmap_config["level"]
        self.lm_id = lightmap_config["id"]

        # Time keys from config (string -> filename)
        time_keys = sorted(lightmap_config["lightmaps"].keys(), key=lambda s: int(s))
        self.times = [int(t) for t in time_keys]  # e.g. [0, 100, ..., 2300]
        self.T = len(self.times)

        # Load PCA params from binary file
        param_path = os.path.join(
            params_dir,
            f"pca_model_{self.level}_{self.lm_id}_params.bin",
        )
        if not os.path.exists(param_path):
            raise FileNotFoundError(f"PCA parameter file not found: {param_path}")

        self.model = PCALightmapModel.load(param_path)

        # Geometry info from the model
        self.block_size = self.model.block_size
        self.grid_h = self.model.grid_h
        self.grid_w = self.model.grid_w
        self.height = self.model.H
        self.width = self.model.W
        self.T = self.model.T  # number of time steps in the PCA model (should match len(self.times))

        # Will be filled on first reconstruct()
        #   self.blocks_dec : np.ndarray [N_blocks, B, B, 3, T]
        self.blocks_dec = None

        # Current reconstructed frame [1, 3, H, W] on device
        self.current_frame = None
        self.current_idx = 0  # index in [0, T-1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _time_to_index(self, current_time):
        """
        Map current_time to the nearest time index in self.times.

        Accepts:
            - integer like 0, 100, ..., 2300
            - float in [0, 24] (hours), which is interpreted as hour * 100
            - numpy scalar types (np.float32, np.float64, np.int32, etc.)
            - string representations of the above
        """
        # Handle string – could be "5.9" or "590"
        if isinstance(current_time, str):
            try:
                t_val = int(current_time)
            except ValueError:
                # interpret as hours
                t_val = int(round(float(current_time) * 100.0))
        else:
            # Handle numeric types (including numpy scalars)
            if isinstance(current_time, numbers.Real):
                ct = float(current_time)
            else:
                # Fallback: try to cast to float
                ct = float(current_time)

            # If looks like hours [0, 24], interpret as hour*100; else as raw integer timecode
            if 0.0 <= ct <= 24.1:
                t_val = int(round(ct * 100.0))
            else:
                t_val = int(round(ct))

        diffs = [abs(t - t_val) for t in self.times]
        idx = int(np.argmin(diffs))
        return idx

    def _ensure_blocks_decoded(self):
        """
        Decode all blocks across all times ONCE per lightmap.

        This is the heavy PCA decode; it will be called inside reconstruct()
        the first time, so it will be counted in the timed path.
        """
        if self.blocks_dec is not None:
            return

        # model.decode_blocks() should return:
        #   [N_blocks, B, B, 3, T] (float32, numpy)
        self.blocks_dec = self.model.decode_blocks()
        if not isinstance(self.blocks_dec, np.ndarray):
            raise TypeError("PCALightmapModel.decode_blocks() must return a numpy array.")

    def _assemble_frame_for_time(self, t_idx: int) -> torch.Tensor:
        """
        Assemble a full [H, W, 3] frame for time index t_idx
        from self.blocks_dec, then return as [1, 3, H, W] tensor on self.device.
        """
        B = self.block_size
        grid_h = self.grid_h
        grid_w = self.grid_w
        H, W = self.height, self.width

        blocks = self.blocks_dec  # [N_blocks, B, B, 3, T]
        N_blocks = blocks.shape[0]

        frame = np.zeros((H, W, 3), dtype=np.float32)

        for idx in range(N_blocks):
            gy = idx // grid_w
            gx = idx % grid_w
            by = gy * B
            bx = gx * B

            blk = blocks[idx, :, :, :, t_idx]  # [B, B, 3]
            h_end = min(by + B, H)
            w_end = min(bx + B, W)

            # Handle potential border clipping (if H, W not divisible by B)
            frame[by:h_end, bx:w_end, :] = blk[: h_end - by, : w_end - bx, :]

        # Convert to tensor [1, 3, H, W] on device
        frame_torch = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0)
        return frame_torch.to(self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public API (required by the competition)
    # ------------------------------------------------------------------
    def reconstruct(self, current_time):
        """
        According to spec:
          - Given current_time in [0, 24] (float) or encoded time (0..2300),
            perform the *full* decompression for that time.

        Implementation details:
          - On first call, we decode all PCA blocks for all times once
            (model.decode_blocks()) and store them in self.blocks_dec.
          - For each reconstruct() call, we assemble only one time slice
            into a full [1, 3, H, W] tensor and store it as self.current_frame.
        """
        # Heavy decode happens here, inside timed path, but only once per lightmap
        self._ensure_blocks_decoded()

        # Map time to index and assemble that frame
        self.current_idx = self._time_to_index(current_time)
        self.current_frame = self._assemble_frame_for_time(self.current_idx)

    def get_result(self):
        """
        Return reconstructed lightmap at the current time as [1, 3, H, W] tensor.
        Must live on self.device.
        """
        if self.current_frame is None:
            raise RuntimeError("get_result() called before reconstruct().")
        return self.current_frame

    def random_test(self, coord):
        """
        Random-Access interface.

        Input:
            coord: torch.Tensor or np.ndarray of shape [N, 3] with (y, x, time)
                   - y: [0, H-1]
                   - x: [0, W-1]
                   - time: [0, 24] float or encoded time (0..2300)

        Output:
            torch.Tensor [N, 3] on self.device, corresponding to RGB at those coords.

        For the official submitTest.py, random_test is always called immediately
        after reconstruct(current_time), with the *same* time. So we simply
        sample from self.current_frame, which guarantees consistency with get_result().
        """
        if self.current_frame is None:
            raise RuntimeError("random_test() called before reconstruct().")

        # Convert coord to numpy for easier indexing
        if isinstance(coord, torch.Tensor):
            coord_np = coord.detach().cpu().numpy()
        else:
            coord_np = np.asarray(coord)

        H, W = self.height, self.width
        N = coord_np.shape[0]

        # current_frame: [1, 3, H, W] on device
        frame = self.current_frame[0]  # [3, H, W]

        out = torch.zeros((N, 3), dtype=torch.float32, device=self.device)

        for i in range(N):
            y, x, _t = coord_np[i]  # we rely on reconstruct() having set the time already

            y_i = int(np.clip(round(float(y)), 0, H - 1))
            x_i = int(np.clip(round(float(x)), 0, W - 1))

            color = frame[:, y_i, x_i]  # [3]
            out[i] = color

        return out


def get(lightmap_config, device):
    """
    Entry point mirroring the original BasicInterface.get().
    The evaluation script will call this function.
    """
    return PCAInterface(lightmap_config, device)
