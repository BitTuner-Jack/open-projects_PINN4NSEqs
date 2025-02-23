import os
import h5py
import numpy as np
import torch
from dataset import WindFluidDataset
from models import PhysicsInformedNN
from visualization import plot_solution_comparison, plot_solution_single
import argparse

def load_data(file_path):
    """Load data from HDF5 file.
    
    Args:
        file_path: Path to the HDF5 data file
        
    Returns:
        Tuple of arrays (x, y, z, t, u, v, w)
    """
    with h5py.File(file_path, 'r') as file:
        data = file['data_uvw'][:].T
        
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    z = data[:, 2].reshape(-1, 1)
    t = data[:, 3].reshape(-1, 1)
    u = data[:, 4].reshape(-1, 1)
    v = data[:, 5].reshape(-1, 1)
    w = data[:, 6].reshape(-1, 1)
    
    return x, y, z, t, u, v, w

def prepare_training_data(x, y, z, t, u, v, w, N_train):
    """Prepare training data by random sampling.
    
    Args:
        x, y, z, t, u, v, w: Full data arrays
        N_train: Number of training points
        
    Returns:
        Training data arrays
    """
    NT = x.shape[0]
    idx = np.random.choice(NT, N_train, replace=False)
    return [arr[idx] for arr in [x, y, z, t, u, v, w]]

def train(args):
    """Training function.
    
    Args:
        args: Command line arguments
    """
    # Load and prepare data
    x, y, z, t, u, v, w = load_data(args.data_path)
    train_data = prepare_training_data(x, y, z, t, u, v, w, args.N_train)
    
    # Convert to tensors
    train_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in train_data]
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhysicsInformedNN(*train_tensors, args.layers, device)
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Train model
    model.train_model(args.nIter, args.save_path)

def test(args):
    """Testing function.
    
    Args:
        args: Command line arguments
    """
    # Load data
    x, y, z, t, u, v, w = load_data(args.data_path)
    
    # Initialize model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhysicsInformedNN(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(z, dtype=torch.float32),
        torch.tensor(t, dtype=torch.float32),
        torch.tensor(u, dtype=torch.float32),
        torch.tensor(v, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
        args.layers,
        device
    )
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Get plane data for visualization
    dataset = WindFluidDataset(x, y, z, t, u, v, w)
    plane_data = dataset.get_plane_data(np.concatenate([x,y,z,t,u,v,w], axis=1), 
                                      args.plane_type,
                                      args.fixed_val,
                                      args.test_time)
    
    # Make predictions
    if args.plane_type == 'XY':
        coords = plane_data[:, :2]
        x_star = coords[:, 0:1]
        y_star = coords[:, 1:2]
        z_star = np.full_like(x_star, args.fixed_val)
    elif args.plane_type == 'XZ':
        coords = plane_data[:, [0,2]]
        x_star = coords[:, 0:1]
        z_star = coords[:, 1:2]
        y_star = np.full_like(x_star, args.fixed_val)
    else:  # YZ
        coords = plane_data[:, 1:3]
        y_star = coords[:, 0:1]
        z_star = coords[:, 1:2]
        x_star = np.full_like(y_star, args.fixed_val)
        
    t_star = np.full_like(x_star, args.test_time)
    
    u_pred, v_pred, w_pred = model.predict(x_star, y_star, z_star, t_star)
    
    # Compute velocities
    from utils import calc_v_magnitude
    speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)
    speed_true = calc_v_magnitude(plane_data[:, 2:3], plane_data[:, 3:4], plane_data[:, 4:5])
    
    # Plot results
    plot_solution_comparison(coords, speed_pred, speed_true, args.plane_type, 
                           os.path.join(args.save_path, f'{args.plane_type}_comparison.png'))
    plot_solution_single(coords, np.abs(speed_true - speed_pred), args.plane_type,
                        os.path.join(args.save_path, f'{args.plane_type}_error.png'))

def main():
    parser = argparse.ArgumentParser(description='PINN for Navier-Stokes')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    parser.add_argument('--data_path', type=str, default='data_uvw.mat')
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--N_train', type=int, default=19700)
    parser.add_argument('--nIter', type=int, default=500000)
    parser.add_argument('--layers', type=int, nargs='+', default=[4, 100, 100, 100, 100, 4])
    
    # Test specific arguments
    parser.add_argument('--model_path', type=str, help='Path to model weights')
    parser.add_argument('--plane_type', type=str, choices=['XY', 'XZ', 'YZ'], default='XZ')
    parser.add_argument('--fixed_val', type=float, help='Fixed value for the third dimension')
    parser.add_argument('--test_time', type=float, help='Time point for testing')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        if not all([args.model_path, args.fixed_val, args.test_time]):
            parser.error('Test mode requires --model_path, --fixed_val, and --test_time')
        test(args)

if __name__ == '__main__':
    main() 