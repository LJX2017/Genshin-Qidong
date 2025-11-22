import torch
import os
from PCA_VQ_Model import PCA_VQ_Model

PARAMETERS = './pca_vq_parameters'

class BasicInterface:
    def __init__(self, lightmap_config, device: torch.device):
        self.device = device

        lm_id = f"{lightmap_config['level']}_{lightmap_config['id']}"
        param_path = os.path.join("pca_vq_parameters", f"vq_{lm_id}.bin")

        # Load using the raw binary loader
        self.model = PCA_VQ_Model.load_raw(param_path, device)

    def reconstruct(self, current_time):
        self.model.reconstruct(current_time)

    def get_result(self):
        return self.model.get_result()

    def random_test(self, coord):
        return self.model.random_test(coord)


def get(lightmap_config, device: torch.device):
    return BasicInterface(lightmap_config, device)