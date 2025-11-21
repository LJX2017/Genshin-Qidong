import torch
import torch.nn as nn
import math


class ExampleModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=256):
        """
        Replacment Model: Fourier Feature Network (Positional Encoding)
        Args:
            input_dim: Expected to be 3 for (x, y, time)
            output_dim: 3 for (r, g, b)
        """
        super(ExampleModel, self).__init__()

        # --- Configuration ---
        # Higher frequencies for spatial coords (x,y) to capture sharp texture details
        self.pos_freqs = 10  # 2^0 ... 2^9
        # Lower frequencies for time (t) as lighting changes are generally smooth
        self.time_freqs = 6  # 2^0 ... 2^5

        # --- Input Dimension Calculation ---
        # Spatial (x, y): 2 coords * (2 * pos_freqs + 1) [+1 for raw input]
        dim_spatial = 2 * (2 * self.pos_freqs + 1)
        # Time (t): 1 coord * (2 * time_freqs + 1)
        dim_time = 1 * (2 * self.time_freqs + 1)
        # Light Switch Status (Binary Feature): 1 dim
        dim_light = 1

        total_input_dim = dim_spatial + dim_time + dim_light

        # --- MLP Backbone ---
        # A deeper network (6-8 layers) is standard for Neural Fields
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Force output to valid RGB range [0, 1]
        )

        # Initialize weights (Crucial for convergence)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def positional_encoding(self, x, num_freqs):
        """
        Applies sin/cos encoding to input x.
        x shape: [Batch, 1]
        Output shape: [Batch, 2 * num_freqs + 1]
        """
        # Keep raw input
        encoded = [x]
        # Create frequencies on the device of x
        freq_bands = 2.0 ** torch.linspace(0., num_freqs - 1, num_freqs, device=x.device)

        for freq in freq_bands:
            # x * pi * freq
            encoded.append(torch.sin(x * freq * math.pi))
            encoded.append(torch.cos(x * freq * math.pi))

        return torch.cat(encoded, dim=-1)

    def forward(self, x):
        """
        x: Tensor of shape [Batch, 3] containing (coord_x, coord_y, time)
           Assumes inputs are normalized in range [0, 1].
        """
        # 1. Split Inputs
        # x[:, 0] -> x coord, x[:, 1] -> y coord, x[:, 2] -> time
        pos_x = x[:, 0:1]
        pos_y = x[:, 1:2]
        time_t = x[:, 2:3]

        # 2. Compute Light Status (Hard feature for the switch)
        # Rules: Off between 6.0 and 18.0.
        # Input time is normalized [0, 1] corresponding to [0, 24h]
        # 6.0/24.0 = 0.25, 18.0/24.0 = 0.75
        # 1.0 if ON, 0.0 if OFF
        is_light_on = 1.0 - ((time_t >= 0.25) & (time_t <= 0.75)).float()

        # 3. Apply Positional Encoding
        x_emb = self.positional_encoding(pos_x, self.pos_freqs)
        y_emb = self.positional_encoding(pos_y, self.pos_freqs)
        t_emb = self.positional_encoding(time_t, self.time_freqs)

        # 4. Concatenate All Features
        # [Batch, dim_spatial + dim_time + dim_light]
        features = torch.cat([x_emb, y_emb, t_emb, is_light_on], dim=-1)

        # 5. MLP Pass
        rgb = self.net(features)

        return rgb