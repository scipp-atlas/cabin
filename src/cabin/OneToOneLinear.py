from __future__ import annotations

import torch
import torch.nn
from torch import Tensor
from torch.nn.parameter import Parameter


class OneToOneLinear(torch.nn.Module):

    __constants__ = ["features"]
    features: int
    weight: Tensor

    def __init__(
        self,
        features: int,
        scalefactor=None,
        weights=None,
        postroot=None,
        device=None,
        dtype=None,
        mode="normal",
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.mode = mode
        
        # Handle window modes by duplicating features
        if mode == "window_keep":
            # Mode 3: Learn window to keep signal (x1 < x < x2)
            self.effective_features = features * 2
            self.original_features = features
            # Fixed weights: [+1, -1] pairs for each feature
            window_weights = []
            for _ in range(features):
                window_weights.extend([1.0, -1.0])  # lower bound (x > cut1), upper bound (x < cut2)
            weights = window_weights
        elif mode == "window_reject":
            # Mode 4: Learn window to reject background (x < x1 || x > x2)
            self.effective_features = features * 2  
            self.original_features = features
            # Will be handled by dual network structure
            window_weights = []
            for _ in range(features):
                window_weights.extend([1.0, -1.0])
            weights = window_weights
        else:
            # Normal mode
            self.effective_features = features
            self.original_features = features
        
        self.features = self.effective_features
        self.trainable_weights = weights is None or len(weights) != self.effective_features
        if self.trainable_weights:
            if weights is None:
                print(
                    "This network will learn weights and biases, since no weights were supplied."
                )
            elif len(weights) != self.effective_features:
                print(
                    f"This network will learn weights and biases, since num weights ({len(weights)}) does not match num features ({self.effective_features})."
                )
            self.weight = Parameter(torch.empty(self.effective_features, **factory_kwargs))
        else:
            self.register_buffer("weight", torch.tensor(weights))
        self.bias = Parameter(torch.empty(self.effective_features, **factory_kwargs))
        self.activation_scale_factor = scalefactor
        self.post_product_root = postroot if postroot is not None else 1.0
        self.reset_parameters()

    def _expand_inputs_for_windows(self, inputs: Tensor) -> Tensor:
        """Expand inputs by duplicating each feature for window learning"""
        # inputs shape: (batch_size, original_features)
        # output shape: (batch_size, original_features * 2)
        batch_size = inputs.shape[0]
        expanded = torch.zeros(batch_size, self.effective_features, device=inputs.device, dtype=inputs.dtype)
        
        for i in range(self.original_features):
            # Each feature gets duplicated: feature i goes to positions 2*i and 2*i+1
            expanded[:, 2*i] = inputs[:, i]      # lower bound cut
            expanded[:, 2*i+1] = inputs[:, i]    # upper bound cut
        
        return expanded

    def reset_parameters(self) -> None:
        if self.trainable_weights:
            bound = 1.0 / (self.effective_features**0.5)
            torch.nn.init.uniform_(self.weight, -bound, bound)
        torch.nn.init.zeros_(self.bias)

    def get_cuts(self) -> Tensor:
        cuts = -self.bias / self.weight
        if self.mode == "window_keep":
            # Return pairs of (lower, upper) cuts for each original feature
            return cuts.view(self.original_features, 2)
        elif self.mode == "window_reject":
            # Return pairs of (lower, upper) cuts for rejection windows
            return cuts.view(self.original_features, 2)
        else:
            return cuts

    def apply_cuts(self, inputs: Tensor) -> Tensor:
        if self.mode in ["window_keep", "window_reject"]:
            # Duplicate inputs for window modes
            expanded_inputs = self._expand_inputs_for_windows(inputs)
            targets = torch.matmul(expanded_inputs, torch.diag(self.weight)) + self.bias
        else:
            # need to turn the "weights" vector into a matrix with the vector
            # elements on the diagonal, and zeroes everywhere else.
            targets = torch.matmul(inputs, torch.diag(self.weight)) + self.bias
        return targets

    def pass_cuts(self, inputs: Tensor) -> Tensor:
        t = self.apply_cuts(inputs)
        return torch.all(t > 0, dim=1)

    def forward(self, inputs: Tensor) -> Tensor:
        # apply the cuts
        targets = self.apply_cuts(inputs)

        # activation function
        targets = torch.sigmoid(self.activation_scale_factor * targets)

        # optionally take the root of the targets
        # had thought this might be a good way to reduce bias introduced
        # by taking the product of so many sigmoids, but this destroys
        # the good agreement we see between the "actual" efficiency
        # and the efficiency we calculate in the loss function.
        # so, for now, make sure post_product_root remains 1.0.
        targets = targets ** (1.0 / self.post_product_root)

        # this is differentiable, unlike using torch.all(torch.gt()) or something else that yields booleans.
        # will converge to 1 for things that pass all cuts, and to zero for things that fail any single cut.
        targets = torch.prod(targets, dim=1)

        return targets

    def extra_repr(self) -> str:
        return f"in_features={self.features}, bias={self.bias}"


class DualNetworkLinear(torch.nn.Module):
    """
    Mode 4: Learn rejection windows by using dual networks
    First network learns to keep signal, second learns to reject background
    """
    
    def __init__(
        self,
        features: int,
        scalefactor=None,
        postroot=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.features = features
        
        # Network 1: Learn windows to keep signal (normal targets)
        self.accept_net = OneToOneLinear(
            features, scalefactor, None, postroot, device, dtype, mode="window_keep"
        )
        
        # Network 2: Learn windows to reject background (flipped targets)  
        self.reject_net = OneToOneLinear(
            features, scalefactor, None, postroot, device, dtype, mode="window_keep"
        )
    
    def get_cuts(self) -> tuple:
        """Return (accept_cuts, reject_cuts) as tuples of (lower, upper) bounds"""
        accept_cuts = self.accept_net.get_cuts()
        reject_cuts = self.reject_net.get_cuts() 
        return accept_cuts, reject_cuts
    
    def forward(self, inputs: Tensor) -> Tensor:
        # Get acceptance probability (signal-focused)
        accept_prob = self.accept_net(inputs)
        
        # Get rejection probability (background-focused, targets will be flipped in loss)
        reject_prob = self.reject_net(inputs)
        
        # Final probability: accept AND not reject
        # reject_prob represents P(should_reject), so (1-reject_prob) is P(should_keep)
        return accept_prob * (1.0 - reject_prob)
