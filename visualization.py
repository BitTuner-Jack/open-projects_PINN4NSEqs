import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_solution_single(X_star, u_star, plane_type='XY', save_path=None):
    """Plot single solution on specified plane.
    
    Args:
        X_star: Location coordinates, shape (N, 2)
        u_star: Solution values, shape (N, 1) 
        plane_type: Type of plane to plot on, one of ['XY', 'XZ', 'YZ']
        save_path: Path to save the plot, if None will not save
    """
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    
    if plane_type == 'XY':
        x = np.linspace(lb[0], ub[0], nn)
        y = np.linspace(lb[1], ub[1], nn)
        X, Y = np.meshgrid(x, y)
        U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    elif plane_type == 'XZ':
        x = np.linspace(lb[0], ub[0], nn) 
        z = np.linspace(lb[1], ub[1], nn)
        X, Z = np.meshgrid(x, z)
        U_star = griddata(X_star, u_star.flatten(), (X, Z), method='cubic')
    else: # YZ
        y = np.linspace(lb[0], ub[0], nn)
        z = np.linspace(lb[1], ub[1], nn)
        Y, Z = np.meshgrid(y, z)
        U_star = griddata(X_star, u_star.flatten(), (Y, Z), method='cubic')

    plt.figure()
    plt.pcolor(X, Y if plane_type=='XY' else Z, U_star, cmap='jet')
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_solution_comparison(X_star, u_pred, u_true, plane_type='XY', save_path=None):
    """Plot predicted vs true solutions side by side.
    
    Args:
        X_star: Location coordinates, shape (N, 2)
        u_pred: Predicted solution values, shape (N, 1)
        u_true: True solution values, shape (N, 1)
        plane_type: Type of plane to plot on, one of ['XY', 'XZ', 'YZ']
        save_path: Path to save the plot, if None will not save
    """
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    vmin = min(u_pred.min(), u_true.min())
    vmax = max(u_pred.max(), u_true.max())
    
    if plane_type == 'XY':
        x = np.linspace(lb[0], ub[0], nn)
        y = np.linspace(lb[1], ub[1], nn)
        X, Y = np.meshgrid(x, y)
        U_pred = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
        U_true = griddata(X_star, u_true.flatten(), (X, Y), method='cubic')
        im1 = ax1.pcolor(X, Y, U_pred, cmap='jet', vmin=vmin, vmax=vmax)
        im2 = ax2.pcolor(X, Y, U_true, cmap='jet', vmin=vmin, vmax=vmax)
    elif plane_type == 'XZ':
        x = np.linspace(lb[0], ub[0], nn)
        z = np.linspace(lb[1], ub[1], nn)
        X, Z = np.meshgrid(x, z)
        U_pred = griddata(X_star, u_pred.flatten(), (X, Z), method='cubic')
        U_true = griddata(X_star, u_true.flatten(), (X, Z), method='cubic')
        im1 = ax1.pcolor(X, Z, U_pred, cmap='jet', vmin=vmin, vmax=vmax)
        im2 = ax2.pcolor(X, Z, U_true, cmap='jet', vmin=vmin, vmax=vmax)
    else: # YZ
        y = np.linspace(lb[0], ub[0], nn)
        z = np.linspace(lb[1], ub[1], nn)
        Y, Z = np.meshgrid(y, z)
        U_pred = griddata(X_star, u_pred.flatten(), (Y, Z), method='cubic')
        U_true = griddata(X_star, u_true.flatten(), (Y, Z), method='cubic')
        im1 = ax1.pcolor(Y, Z, U_pred, cmap='jet', vmin=vmin, vmax=vmax)
        im2 = ax2.pcolor(Y, Z, U_true, cmap='jet', vmin=vmin, vmax=vmax)
    
    ax1.set_title('Predicted')
    ax2.set_title('True')
    fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_error_distribution(values, errors, xlabel, save_path=None):
    """Plot error distribution along a dimension.
    
    Args:
        values: Values along the dimension
        errors: Corresponding error values
        xlabel: Label for x-axis
        save_path: Path to save the plot, if None will not save
    """
    plt.figure()
    plt.plot(values, errors, marker='o', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel('Error')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show() 