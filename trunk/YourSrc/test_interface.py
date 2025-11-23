import os
from typing import Dict

import numpy as np
import torch

from test_model import PCALightmapModel, encode_time_value, prepare_time_lookup


class BasicInterface:
    def __init__(self, lightmap_config, device, param_dir: str = "./test_parameters"):
        self.device = device
        self.lightmap_config = lightmap_config
        self.param_dir = param_dir
        self.id = f"{lightmap_config['level']}_{lightmap_config['id']}"

        param_path = os.path.join(self.param_dir, f"pca_{self.id}.npz")
        if not os.path.exists(param_path):
            raise FileNotFoundError(f"PCA parameter file not found: {param_path}")

        model = PCALightmapModel.from_npz(param_path)
        self.height, self.width = model.resolution
        self.time_lookup: Dict[int, int] = prepare_time_lookup(model.time_tokens)
        self.time_count = len(model.time_tokens)

        self.cluster_data = []
        self.cluster_lookup = -np.ones(self.height * self.width, dtype=np.int32)
        self.row_lookup = -np.ones(self.height * self.width, dtype=np.int32)

        for cluster_idx, cluster in enumerate(model.clusters):
            n_comp = cluster.component_count
            components = torch.from_numpy(cluster.components.reshape(n_comp, self.time_count, 3)).to(self.device)
            mean = torch.from_numpy(cluster.mean.reshape(self.time_count, 3)).to(self.device)
            coeffs = torch.from_numpy(cluster.coeffs).to(self.device)
            if coeffs.ndim == 1:
                coeffs = coeffs.view(-1, 1)
            pixel_indices = torch.from_numpy(cluster.pixel_indices).long().to(self.device)

            self.cluster_data.append(
                {
                    "components": components,
                    "mean": mean,
                    "coeffs": coeffs,
                    "pixel_indices": pixel_indices,
                    "pixel_count": coeffs.shape[0],
                    "component_count": n_comp,
                }
            )

            pix_cpu = cluster.pixel_indices.astype(np.int64)
            self.cluster_lookup[pix_cpu] = cluster_idx
            self.row_lookup[pix_cpu] = np.arange(len(pix_cpu), dtype=np.int32)

        self.result = None

    def _token_for_time(self, current_time: float) -> int:
        token = encode_time_value(current_time)
        if token not in self.time_lookup:
            raise ValueError(f"Unsupported time {current_time} (token {token}) for {self.id}")
        return token

    def reconstruct(self, current_time: float):
        token = self._token_for_time(current_time)
        time_idx = self.time_lookup[token]

        flat = torch.zeros((self.height * self.width, 3), device=self.device)
        for cluster in self.cluster_data:
            basis = cluster["components"][:, time_idx, :]
            base = cluster["mean"][time_idx, :].unsqueeze(0)
            values = torch.matmul(cluster["coeffs"], basis) + base
            flat[cluster["pixel_indices"]] = values

        image = flat.view(self.height, self.width, 3).permute(2, 0, 1)
        self.result = image.unsqueeze(0)

    def get_result(self):
        if self.result is None:
            raise RuntimeError("Call reconstruct() before get_result().")
        return self.result

    def random_test(self, coord):
        if isinstance(coord, torch.Tensor):
            coord_np = coord.detach().cpu().numpy()
        else:
            coord_np = np.asarray(coord)

        y = int(coord_np[0, 0])
        x = int(coord_np[0, 1])
        token = self._token_for_time(float(coord_np[0, 2]))
        time_idx = self.time_lookup[token]

        flat_index = y * self.width + x
        cluster_idx = self.cluster_lookup[flat_index]
        if cluster_idx < 0:
            return torch.zeros((1, 3), device=self.device)
        row_idx = int(self.row_lookup[flat_index])

        cluster = self.cluster_data[cluster_idx]
        basis = cluster["components"][:, time_idx, :]
        base = cluster["mean"][time_idx, :].unsqueeze(0)
        coeff_vec = cluster["coeffs"][row_idx : row_idx + 1, :]
        value = torch.matmul(coeff_vec, basis) + base
        return value


def get(lightmap_config, device, param_dir: str = "./test_parameters"):
    return BasicInterface(lightmap_config, device, param_dir=param_dir)
