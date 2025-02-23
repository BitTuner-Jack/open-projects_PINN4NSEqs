import torch
from torch.utils.data import Dataset
import numpy as np

class WindFluidDataset(Dataset):
    """Dataset class for wind fluid data.
    
    Attributes:
        x: x coordinates
        y: y coordinates 
        z: z coordinates
        t: time points
        u: u velocity component
        v: v velocity component
        w: w velocity component
        phi: phi angle (optional)
        theda: theda angle (optional)
    """
    
    def __init__(self, x, y, z, t, u, v, w, phi=None, theda=None):
        """Initialize dataset.
        
        Args:
            x: x coordinates, shape (N, 1)
            y: y coordinates, shape (N, 1)
            z: z coordinates, shape (N, 1)
            t: time points, shape (N, 1)
            u: u velocity, shape (N, 1)
            v: v velocity, shape (N, 1)
            w: w velocity, shape (N, 1)
            phi: phi angle, shape (N, 1), optional
            theda: theda angle, shape (N, 1), optional
        """
        super().__init__()
        self.x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        self.y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
        self.z = torch.tensor(z, dtype=torch.float32, requires_grad=True)
        self.t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32)
        self.v = torch.tensor(v, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)
        
        if phi is not None and theda is not None:
            self.phi = torch.tensor(phi, dtype=torch.float32)
            self.theda = torch.tensor(theda, dtype=torch.float32)
        else:
            self.phi = None
            self.theda = None
            
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        data = (self.x[idx], self.y[idx], self.z[idx], self.t[idx],
                self.u[idx], self.v[idx], self.w[idx])
        if self.phi is not None:
            data += (self.phi[idx], self.theda[idx])
        return data
    
    @staticmethod
    def get_plane_data(data, plane_type, fixed_val, t):
        """Extract data on a specific plane.
        
        Args:
            data: Full dataset array
            plane_type: One of ['XY', 'XZ', 'YZ']
            fixed_val: Fixed value for the third dimension
            t: Time point
            
        Returns:
            Plane data array
        """
        if plane_type == 'XY':
            return data[(data[:, 2] == fixed_val) & (data[:, 3] == t)][:, [0,1,4,5,6]]
        elif plane_type == 'XZ':
            return data[(data[:, 1] == fixed_val) & (data[:, 3] == t)][:, [0,2,4,5,6]]
        else: # YZ
            return data[(data[:, 0] == fixed_val) & (data[:, 3] == t)][:, [1,2,4,5,6]]
    


