# Physics-Informed Neural Networks for Navier-Stokes Equations （Wind-Filed Reconstruction）

This project implements a Physics-Informed Neural Network (PINN) to solve the 3D Navier-Stokes equations. The model combines data observations with physical constraints to predict fluid dynamics.

## Features

- 3D Navier-Stokes equations solver using PINN
- Automatic differentiation for computing spatial and temporal derivatives
- Reynolds number optimization during training
- Real-time visualization of training progress
- Multiple visualization options for results (XY, XZ, YZ planes)

## Project Structure 
```bash
├── dataset.py # Data loading and preprocessing
├── models.py # PINN model implementation
├── utils.py # Utility functions
├── visualization.py # Visualization functions
├── main.py # Training and testing scripts
└── results/ # Training results and visualizations
```

## Usage

Training the model:
```bash
bash
python main.py --mode train \
--data_path data_uvw.mat \
--save_path results \
--N_train 19700 \
--nIter 500000
```

Testing the model:
```bash
bash
python main.py --mode test \
--data_path data_uvw.mat \
--model_path results/PINN_500000.pth \
--plane_type XZ \
--fixed_val 330 \
--test_time 100
```


## Results

### Velocity Field Prediction

Below shows the comparison between predicted and true velocity fields on the XY plane:

![Velocity Field Comparison](results/X-Y_filed.png)

### Error Distribution

The absolute error distribution between predicted and true values:

<img src="results/X-Y_diff.png" alt="Error Distribution" style="zoom: 67%;" /> <img src="results/X-Y.png" alt="Err" style="zoom: 67%;" />



## Model Architecture

- Input layer: 4 neurons (x, y, z, t)
- Hidden layers: 4 layers with 100 neurons each
- Output layer: 4 neurons (u, v, w, p)
- Activation function: tanh
- Loss function: MSE of velocity magnitude + Physics-informed constraints

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

## License

MIT License
