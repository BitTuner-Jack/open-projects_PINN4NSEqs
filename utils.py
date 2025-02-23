import numpy as np
import torch

def calc_v_magnitude(u, v, w):
    """Calculate velocity magnitude from components.
    
    Args:
        u: u velocity component
        v: v velocity component
        w: w velocity component
        
    Returns:
        Velocity magnitude
    """
    if isinstance(u, torch.Tensor):
        return torch.sqrt(u**2 + v**2 + w**2)
    return np.sqrt(u**2 + v**2 + w**2)

def calc_v_direction(u, v, w, phi, theda):
    """Calculate directional velocity.
    
    Args:
        u: u velocity component
        v: v velocity component
        w: w velocity component
        phi: phi angle in degrees
        theda: theda angle in degrees
        
    Returns:
        Directional velocity
    """
    if isinstance(u, torch.Tensor):
        theda_rad = theda * torch.pi / 180
        phi_rad = phi * torch.pi / 180
    else:
        theda_rad = theda * np.pi / 180
        phi_rad = phi * np.pi / 180
        
    v_r = (u * np.cos(theda_rad) * np.cos(phi_rad) + 
           v * np.sin(theda_rad) * np.cos(phi_rad) +
           w * np.sin(phi_rad))
    return v_r

def xavier_init(size):
    """Initialize weights using Xavier initialization.
    
    Args:
        size: Tuple of (input_dim, output_dim)
        
    Returns:
        Initialized weights as torch Parameter
    """
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    return torch.nn.Parameter(torch.randn(size) * xavier_stddev)
