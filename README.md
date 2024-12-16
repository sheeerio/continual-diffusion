# Continual Diffusion: Exploration and Adaptation in Non-Stationary Tasks with Diffusion Policies

This repository contains the implementation for the research project **"Exploration and Adaptation in Non-Stationary Tasks with Diffusion Policies"**. The project investigates the application of diffusion models in reinforcement learning (RL) for non-stationary, vision-based tasks. By leveraging the iterative refinement process of diffusion policies, this work addresses the challenges posed by dynamic environments where task objectives and dynamics evolve over time.

## Overview

The project explores the use of **Diffusion Policies**, which use a denoising diffusion probabilistic model (DDPM) to iteratively refine action sequences. This approach is evaluated across three challenging non-stationary environments: 
- **CoinRun**: A procedurally generated 2D platformer.
- **Maze**: A discrete-action navigation task.
- **PointMaze**: A continuous-action planning task.

The results demonstrate superior performance compared to traditional RL algorithms like PPO and DQN in terms of stability and adaptability under changing conditions.

---

## Features

- Implementation of **Diffusion Policies** for reinforcement learning.
- Training and evaluation in **Procgen** and **D4RL** environments.
- A modular framework supporting discrete and continuous action spaces.
- Closed-loop control with iterative feedback for enhanced adaptability.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sheeerio/continual-diffusion.git
   cd continual-diffusion
   ```
2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional)** Set up a GPU-enabled environment for efficient training.

---

## Repository Structure
- `main.py`: Entry point for training and evaluation workflows.
- `data.py`: Handles data loading and preprocessing for all environments.
- `model.py`: Defines the architecture of the Diffusion Policy, including visual encoder and DDPM.
- `train.py`: Implements the training loop, including dataset loading and model updates.
- `data_collection.py`: Scripts for collecting and augmenting trajectories in RL environments.
- `README.md`: Documentation for the repository.

---

## Usage

### Training a Diffusion Policy
To train a model in a specific environment:
```bash
python train.py --env CoinRun --epochs 100 --batch_size 64
```

### Evaluation
Evaluate the trained model on test episodes:
```bash
python main.py --env CoinRun --eval --model_path <path_to_model>
```

### Data Collection
Generate and preprocess data for training:
```bash
python data_collection.py --env CoinRun --output_dir ./data
```

---

## Results

### Baseline Performance

The table below summarizes the performance of the Diffusion Policy compared to PPO and DQN across all environments:

| Task       | Algorithm   | Mean Reward | Max Reward | Std Dev |
|------------|-------------|-------------|------------|---------|
| CoinRun    | Diffusion   | 8.15        | 8.30       | 0.15    |
| Maze       | Diffusion   | 9.00        | 9.00       | 0.05    |
| PointMaze  | Diffusion   | 93.50       | 98.50      | 1.55    |

For more details on performance and ablation studies, refer to the Results section in the project report.

---

### Highlights of the Architecture

- **Visual Encoder:**
  - A ResNet-based encoder extracts spatial features from raw visual observations.
  - Support for RGB inputs and temporal stacking for dynamic tasks.
  
- **Diffusion Model:**
  - A conditional U-Net processes noisy action proposals and refines them iteratively.
  - Closed-loop control mechanism for continuous adaptability.
  
- **Unified Observation Representation:**
  - Integration of visual features with low-dimensional state vectors for tasks like PointMaze.

---

### Key Insights

- The Diffusion Policy achieves superior performance in non-stationary tasks with complex visual inputs.
- Iterative denoising enables adaptive planning, especially in dynamically changing environments.
- Challenges include high computational demands and limitations in handling extreme non-stationarity.

---

### References

- Janner, M., Li, Q., and Levine, S. (2022). Planning with Diffusion for Flexible Behavior Synthesis. *ICLR*.
- Chi, L., Ding, M., Lu, Y., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *NeurIPS*.
- Parisi, G.I., et al. (2019). Continual lifelong learning with neural networks: A review. *Neuroscience & Biobehavioral Reviews*.
- For a full list of references, see the project report.

---

*This project was developed for the course [CS533V](https://www.cs.ubc.ca/~van/cpsc533V/index.html).*
