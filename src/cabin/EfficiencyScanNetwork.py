import torch
import torch.nn

from cabin.OneToOneLinear import OneToOneLinear, DualNetworkLinear


class EfficiencyScanNetwork(torch.nn.Module):
    def __init__(
        self, features, effics, weights=None, activationscale=2.0, postroot=1.0, mode="normal"
    ):
        super().__init__()
        self.features = features
        self.effics = effics
        self.weights = weights
        self.activation_scale_factor = activationscale
        self.post_product_root = postroot
        self.mode = mode
        
        # Create networks based on mode
        if mode == "window_reject":
            # Mode 4: Use dual networks for rejection windows
            self.nets = torch.nn.ModuleList(
                [
                    DualNetworkLinear(
                        features,
                        self.activation_scale_factor,
                        self.post_product_root,
                    )
                    for i in range(len(self.effics))
                ]
            )
        else:
            # Modes 1, 2, 3: Use regular OneToOneLinear
            self.nets = torch.nn.ModuleList(
                [
                    OneToOneLinear(
                        features,
                        self.activation_scale_factor,
                        self.weights,
                        self.post_product_root,
                        mode=mode,
                    )
                    for i in range(len(self.effics))
                ]
            )

    def forward(self, x):
        outputs = torch.stack(tuple(self.nets[i](x) for i in range(len(self.effics))))
        return outputs

    def to(self, device):  # noqa: pylint: disable=W0221
        super().to(device)
        for n in self.nets:
            n.to(device)

    def __getitem__(self, key):
        if key < len(self.nets):
            return self.nets[key]
        raise KeyError(f"Key '{key}' not found in nets.")
