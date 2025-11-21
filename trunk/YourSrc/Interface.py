import math
import os
import numpy as np
import torch
import torch.nn as nn


class NeuralLightmap(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=6):
        super().__init__()
        self.pos_freqs = 10
        self.time_freqs = 4
        input_dim = (2 * (2 * self.pos_freqs + 1)) + (2 * self.time_freqs + 1) + 1

        layers = []
        last_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU(inplace=True)])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 3))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def positional_encoding(self, x, num_freqs):
        res = [x]
        freqs = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs, device=x.device)
        for f in freqs:
            res.append(torch.sin(x * f * math.pi))
            res.append(torch.cos(x * f * math.pi))
        return torch.cat(res, dim=-1)

    def forward(self, x, y, t, light_status):
        x_emb = self.positional_encoding(x, self.pos_freqs)
        y_emb = self.positional_encoding(y, self.pos_freqs)
        t_emb = self.positional_encoding(t, self.time_freqs)
        features = torch.cat([x_emb, y_emb, t_emb, light_status], dim=-1)
        return self.net(features)


class BasicInterface:
    def __init__(self, lightmap_config, device):
        self.device = device
        self.model = NeuralLightmap()
        self._load_parameters()
        self.model.eval().to(self.device)

        resolution = lightmap_config['resolution']
        self.resolution = resolution
        self.height = resolution['height']
        self.width = resolution['width']

    def reconstruct(self, current_time):
        H, W = self.height, self.width
        ys, xs = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        y_norm = (ys.reshape(-1, 1) / max(H - 1, 1)).float()
        x_norm = (xs.reshape(-1, 1) / max(W - 1, 1)).float()
        t_norm = torch.full((H * W, 1), float(current_time) / 24.0, device=self.device)
        light_status = torch.full((H * W, 1), self._light_status_scalar(current_time), device=self.device)
        self.result = self.model(x_norm, y_norm, t_norm, light_status)

    def get_result(self):
        return self.result.reshape(self.height, self.width, 3).permute(2, 0, 1).unsqueeze(0)

    def random_test(self, coord):
        coord_tensor = torch.from_numpy(coord).to(torch.float32).to(self.device)
        y_norm = (coord_tensor[:, 0] / max(self.height - 1, 1)).unsqueeze(-1)
        x_norm = (coord_tensor[:, 1] / max(self.width - 1, 1)).unsqueeze(-1)
        times = coord_tensor[:, 2]
        t_norm = (times / 24.0).unsqueeze(-1)
        light_status = self._light_status_tensor(times).unsqueeze(-1)
        return self.model(x_norm, y_norm, t_norm, light_status)

    def _load_parameters(self):
        params_path = "./Parameters/neural_lightmap_params.bin"
        if not os.path.exists(params_path):
            return
        params_array = np.fromfile(params_path, dtype=np.float32)
        param_idx = 0
        with torch.no_grad():
            for param in self.model.parameters():
                param_size = param.numel()
                if param_idx + param_size > params_array.size:
                    break
                param_data = params_array[param_idx:param_idx + param_size]
                param.copy_(torch.from_numpy(param_data.reshape(param.shape)))
                param_idx += param_size

    def _light_status_scalar(self, current_time):
        return 0.0 if 6.0 <= current_time <= 18.0 else 1.0

    def _light_status_tensor(self, times):
        return torch.where((times >= 6.0) & (times <= 18.0),
                           torch.zeros_like(times),
                           torch.ones_like(times))


def get(lightmap_config, device):
    return BasicInterface(lightmap_config, device)
