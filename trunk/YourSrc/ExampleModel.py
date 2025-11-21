import torch
import torch.nn as nn
import tinycudann as tcnn

class ExampleModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=64):
        super(ExampleModel, self).__init__()

        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,  # BASELINE HIGH QUALITY (10MB size)
                "base_resolution": 16,
                "per_level_scale": 1.5
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        }

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=input_dim,
            n_output_dims=output_dim,
            encoding_config=config["encoding"],
            network_config=config["network"]
        )

    def forward(self, x):
        return self.model(x)