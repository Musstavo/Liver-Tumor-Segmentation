import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm


class LiverSegModel(UNet):
    def __init__(self):
        super().__init__(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()
