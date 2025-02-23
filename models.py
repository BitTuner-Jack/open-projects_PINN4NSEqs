import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from utils import xavier_init, calc_v_magnitude

class PhysicsInformedNN(nn.Module):
    """Physics Informed Neural Network for Navier-Stokes equations.
    
    This model implements a neural network that is trained to satisfy both
    the Navier-Stokes equations and data observations.
    """
    
    def __init__(self, x, y, z, t, u, v, w, layers, device):
        """Initialize the PINN model.
        
        Args:
            x: x coordinates, shape (N, 1)
            y: y coordinates, shape (N, 1)
            z: z coordinates, shape (N, 1)
            t: time points, shape (N, 1)
            u: u velocity, shape (N, 1)
            v: v velocity, shape (N, 1)
            w: w velocity, shape (N, 1)
            layers: List of layer sizes
            device: Device to run computations on
        """
        super(PhysicsInformedNN, self).__init__()

        X = torch.cat([x, y, z, t], dim=1)  # N x 4
        self.lb = X.min(0)[0].to(device)
        self.ub = X.max(0)[0].to(device)
        
        # Store coordinates and velocities
        self.x = X[:, 0:1].requires_grad_(True).to(device)
        self.y = X[:, 1:2].requires_grad_(True).to(device)
        self.z = X[:, 2:3].requires_grad_(True).to(device)
        self.t = X[:, 3:4].requires_grad_(True).to(device)
        
        self.u = u.to(device)
        self.v = v.to(device)
        self.w = w.to(device)
    
        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # Initialize Reynolds number
        self.register_parameter("log_Re", nn.Parameter(torch.tensor([6.0], device=device)))  # log10(100) = 2.0
        
        # Setup optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.optimizer_lbfgs = optim.LBFGS(
            self.parameters(), 
            max_iter=50000,
            max_eval=50000,
            tolerance_grad=1e-9,
            tolerance_change=1e-7,
            history_size=50,
            line_search_fn='strong_wolfe'
        )
        
        self.device = device
        
    def initialize_NN(self, layers):
        """Initialize neural network weights and biases.
        
        Args:
            layers: List of layer sizes
            
        Returns:
            weights, biases: Lists of weight and bias Parameters
        """
        weights = nn.ParameterList()
        biases = nn.ParameterList()
        
        for l in range(len(layers)-1):
            weights.append(xavier_init([layers[l], layers[l+1]]))
            biases.append(nn.Parameter(torch.zeros(1, layers[l+1])))
            
        return weights.to(self.device), biases.to(self.device)
    
    def neural_net(self, X):
        """Forward pass through neural network.
        
        Args:
            X: Input tensor, shape (N, 4)
            
        Returns:
            Network output
        """
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        
        for l in range(len(self.weights)-1):
            W = self.weights[l]
            b = self.biases[l]
            H = torch.tanh(torch.addmm(b, H, W))
            
        W = self.weights[-1]
        b = self.biases[-1]
        Y = torch.addmm(b, H, W)
        
        return Y
    
    def net_NS(self, x, y, z, t):
        """Compute Navier-Stokes related quantities.
        
        Args:
            x: x coordinates
            y: y coordinates
            z: z coordinates
            t: time points
            
        Returns:
            u, v, w: Velocity components
            p: Pressure
            f_u, f_v, f_w: Momentum equations residuals
            f_e: Continuity equation residual
        """
        X = torch.cat([x, y, z, t], dim=1)
        outputs = self.neural_net(X)
        u, v, w, p = [outputs[:, i:i+1] for i in range(4)]
        
        # Compute all required derivatives
        derivatives = self.compute_derivatives(u, v, w, p, x, y, z, t)
        
        # Compute equation residuals
        f_u, f_v, f_w, f_e = self.compute_residuals(u, v, w, p, derivatives)
        
        return u, v, w, p, f_u, f_v, f_w, f_e
    
    def compute_derivatives(self, u, v, w, p, x, y, z, t):
        """Compute all required derivatives for Navier-Stokes equations.
        
        Args:
            u, v, w: Velocity components
            p: Pressure
            x, y, z: Spatial coordinates
            t: Time coordinate
            
        Returns:
            Dictionary containing all computed derivatives
        """
        grads = {}
        
        # First derivatives for u
        grads['u_t'] = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grads['u_x'] = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grads['u_y'] = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grads['u_z'] = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # First derivatives for v
        grads['v_t'] = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        grads['v_x'] = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        grads['v_y'] = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        grads['v_z'] = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        
        # First derivatives for w
        grads['w_t'] = torch.autograd.grad(w, t, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        grads['w_x'] = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        grads['w_y'] = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        grads['w_z'] = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        
        # Second derivatives for u
        grads['u_xx'] = torch.autograd.grad(grads['u_x'], x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grads['u_yy'] = torch.autograd.grad(grads['u_y'], y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grads['u_zz'] = torch.autograd.grad(grads['u_z'], z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Second derivatives for v
        grads['v_xx'] = torch.autograd.grad(grads['v_x'], x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        grads['v_yy'] = torch.autograd.grad(grads['v_y'], y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        grads['v_zz'] = torch.autograd.grad(grads['v_z'], z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        
        # Second derivatives for w
        grads['w_xx'] = torch.autograd.grad(grads['w_x'], x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        grads['w_yy'] = torch.autograd.grad(grads['w_y'], y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        grads['w_zz'] = torch.autograd.grad(grads['w_z'], z, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        
        # Pressure derivatives
        grads['p_x'] = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        grads['p_y'] = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        grads['p_z'] = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        return grads
    
    def compute_residuals(self, u, v, w, p, grads):
        """Compute Navier-Stokes equation residuals."""
        # Convert log_Re to actual Re number
        Re = 10.0 ** self.log_Re
        
        # Momentum equations
        f_u = (grads['u_t'] + u * grads['u_x'] + v * grads['u_y'] + w * grads['u_z'] + 
               grads['p_x'] - (1/Re) * (grads['u_xx'] + grads['u_yy'] + grads['u_zz']))
        
        f_v = (grads['v_t'] + u * grads['v_x'] + v * grads['v_y'] + w * grads['v_z'] + 
               grads['p_y'] - (1/Re) * (grads['v_xx'] + grads['v_yy'] + grads['v_zz']))
        
        f_w = (grads['w_t'] + u * grads['w_x'] + v * grads['w_y'] + w * grads['w_z'] + 
               grads['p_z'] - (1/Re) * (grads['w_xx'] + grads['w_yy'] + grads['w_zz']))
        
        # Continuity equation
        f_e = grads['u_x'] + grads['v_y'] + grads['w_z']
        
        return f_u, f_v, f_w, f_e
    
    def forward(self):
        """Forward pass computing the loss."""
        # Optionally add bounds to log_Re if needed
        self.log_Re.data.clamp_(min=0.0, max=10.0)  # Limits Re between 1 and 10^10
        
        u_pred, v_pred, w_pred, _, f_u_pred, f_v_pred, f_w_pred, f_e_pred = self.net_NS(
            self.x, self.y, self.z, self.t)
        
        # Compute velocity magnitudes
        speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)
        speed = calc_v_magnitude(self.u, self.v, self.w)
        
        # Compute losses
        data_loss = torch.mean((speed - speed_pred) ** 2)
        equation_loss = (torch.mean(f_u_pred**2) + torch.mean(f_v_pred**2) + 
                        torch.mean(f_w_pred**2) + torch.mean(f_e_pred**2))
        
        return (data_loss + equation_loss) / 2
    
    def train_model(self, nIter, save_path):
        """Train the model.
        
        Args:
            nIter: Number of iterations
            save_path: Path to save model checkpoints
        """
        writer = SummaryWriter(os.path.join(save_path, 'tensorboard_logs'))
        global_step = 0
        total_loss = 0
        
        def closure():
            self.optimizer.zero_grad()
            loss = self.forward()
            loss.backward()
            return loss
        
        for it in range(nIter):
            start_time = time.time()
            self.optimizer.step(closure)
            
            if (it + 1) % 100 == 0:
                loss_value = self.forward().item()
                Re_value = 10.0 ** self.log_Re.item()
                total_loss += loss_value
                
                elapsed = time.time() - start_time
                print(f'It: {it+1}, Loss: {loss_value:.3f}, Re: {Re_value:.2e}, Time: {elapsed:.2f}')
                
                writer.add_scalar('Loss/iter', loss_value, global_step)
                writer.add_scalar('Params/Re', Re_value, global_step)
                writer.add_scalar('Params/log_Re', self.log_Re.item(), global_step)
                global_step += 100
            
            if (it + 1) % 10000 == 0:
                avg_loss = total_loss / 100
                writer.add_scalar('Loss/epoch', avg_loss, global_step // 10000)
                torch.save(self.state_dict(), os.path.join(save_path, f'PINN_{it+1}.pth'))
                total_loss = 0
                
        print("Training finished!")
        writer.close()
    
    def predict(self, x_star, y_star, z_star, t_star):
        """Make predictions at new points.
        
        Args:
            x_star: x coordinates
            y_star: y coordinates
            z_star: z coordinates
            t_star: time points
            
        Returns:
            u_star, v_star, w_star: Predicted velocities
        """
        # Convert inputs to tensors
        inputs = [torch.tensor(arr, dtype=torch.float32, requires_grad=True, device=self.device)
                 for arr in [x_star, y_star, z_star, t_star]]
        
        # Get predictions
        u_star, v_star, w_star, *_ = self.net_NS(*inputs)
        
        # Convert to numpy
        return [arr.cpu().detach().numpy() for arr in [u_star, v_star, w_star]] 