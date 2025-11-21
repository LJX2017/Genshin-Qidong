import torch
import os
import numpy as np
from ExampleModel import ExampleModel


class BasicInterface:
    def __init__(self, lightmap_config, device):
        self.device = device
        self.model = ExampleModel(input_dim=3, output_dim=3, hidden_dim=64)

        path = f"./Parameters/model_{lightmap_config['level']}_{lightmap_config['id']}_params.bin"

        # BASELINE: Load Float32
        params_array = np.fromfile(path, dtype=np.float32)

        idx = 0
        with torch.no_grad():
            for param in self.model.parameters():
                count = param.numel()
                param.data = torch.from_numpy(params_array[idx: idx + count].reshape(param.shape)).to(device)
                idx += count

        self.model.eval()
        self.resolution = lightmap_config['resolution']

    def reconstruct(self, current_time):
        H, W = self.resolution['height'], self.resolution['width']
        ys, xs = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')

        coords = torch.stack([
            ys.ravel() / (H - 1),
            xs.ravel() / (W - 1),
            torch.full((H * W,), float(current_time), device=self.device) / 24.0
        ], dim=-1).float().contiguous()

        self.result = torch.clamp(torch.expm1(self.model(coords)), min=0.0)

    def get_result(self):
        return self.result.reshape(self.resolution['height'], self.resolution['width'], 3).permute(2, 0, 1).unsqueeze(0)

    def random_test(self, coord):
        c = torch.from_numpy(coord).float().to(self.device)
        c[:, 0] /= (self.resolution['height'] - 1)
        c[:, 1] /= (self.resolution['width'] - 1)
        c[:, 2] /= 24.0
        c = c.contiguous()
        return torch.clamp(torch.expm1(self.model(c)), min=0.0)


def get(lightmap_config, device):
    return BasicInterface(lightmap_config, device)