import torch
import cabin
import pytest


def generate_test_data(N=1000, m=3):
    """Generate simple test data with signal and background"""
    torch.manual_seed(42)
    
    # Signal: higher values in features
    x_signal = torch.randn(N, m) + 1.0
    y_signal = torch.ones(N)
    
    # Background: lower values in features  
    x_backgr = torch.randn(N, m) - 1.0
    y_backgr = torch.zeros(N)
    
    x = torch.concatenate((x_signal, x_backgr))
    y = torch.concatenate((y_signal, y_backgr))
    
    return x.float(), y


def test_window_keep_mode():
    """Test mode 3: automatic window learning to keep signal"""
    x, y = generate_test_data()
    m = x.shape[1]
    
    # Create network with window_keep mode
    efficnet = cabin.EfficiencyScanNetwork(
        m, [0.5], weights=None, activationscale=5.0, mode="window_keep"
    )
    
    # Test forward pass
    y_pred = efficnet(x)
    assert y_pred.shape == (1, len(x))
    
    # Test cuts shape - should be (original_features, 2) for each network
    cuts = efficnet.nets[0].get_cuts()
    assert cuts.shape == (m, 2), f"Expected cuts shape ({m}, 2), got {cuts.shape}"
    
    # Test training step
    optimizer = torch.optim.SGD(efficnet.parameters(), lr=0.1)
    
    loss = cabin.effic_loss_fn(y_pred, y, m, efficnet, alpha=1.0, beta=0.1, gamma=1e-3)
    loss.totalloss().backward()
    optimizer.step()
    
    assert loss.signaleffic > 0
    assert loss.backgreffic >= 0


def test_window_reject_mode():
    """Test mode 4: dual network for rejection windows"""
    x, y = generate_test_data()
    m = x.shape[1]
    
    # Create network with window_reject mode
    efficnet = cabin.EfficiencyScanNetwork(
        m, [0.5], weights=None, activationscale=5.0, mode="window_reject"
    )
    
    # Test forward pass
    y_pred = efficnet(x)
    assert y_pred.shape == (1, len(x))
    
    # Test cuts shape - should return tuple of (accept_cuts, reject_cuts)
    cuts = efficnet.nets[0].get_cuts()
    assert isinstance(cuts, tuple), "Expected tuple of (accept_cuts, reject_cuts)"
    accept_cuts, reject_cuts = cuts
    assert accept_cuts.shape == (m, 2), f"Expected accept cuts shape ({m}, 2)"
    assert reject_cuts.shape == (m, 2), f"Expected reject cuts shape ({m}, 2)"
    
    # Test training step
    optimizer = torch.optim.SGD(efficnet.parameters(), lr=0.1)
    
    loss = cabin.effic_loss_fn(y_pred, y, m, efficnet, alpha=1.0, beta=0.1, gamma=1e-3)
    loss.totalloss().backward()
    optimizer.step()
    
    assert loss.signaleffic > 0
    assert loss.backgreffic >= 0


def test_backward_compatibility():
    """Test that existing modes still work"""
    x, y = generate_test_data()
    m = x.shape[1]
    
    # Test mode 1: learn weights and biases (default)
    efficnet1 = cabin.EfficiencyScanNetwork(m, [0.5], weights=None, activationscale=5.0)
    y_pred1 = efficnet1(x)
    assert y_pred1.shape == (1, len(x))
    
    # Test mode 2: fixed weights, learn biases
    fixed_weights = [1.0] * m
    efficnet2 = cabin.EfficiencyScanNetwork(m, [0.5], weights=fixed_weights, activationscale=5.0)
    y_pred2 = efficnet2(x)
    assert y_pred2.shape == (1, len(x))


if __name__ == "__main__":
    test_window_keep_mode()
    test_window_reject_mode() 
    test_backward_compatibility()
    print("All tests passed!")