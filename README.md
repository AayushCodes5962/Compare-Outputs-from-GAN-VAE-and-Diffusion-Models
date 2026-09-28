# VAE, GAN and Diffusion Models with PyTorch

A practical exploration and comparison of three important generative modeling approaches: Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), and Diffusion Models.

This project uses PyTorch to implement each model from scratch and demonstrates different applications of generative modeling, including anomaly detection, synthetic data generation, class balancing, and patient trajectory generation.

## Overview

Generative models learn the underlying patterns of data and can be used to reconstruct existing samples or generate new synthetic samples.

This project explores three different approaches:

1. VAE for anomaly detection using ECG-like signals
2. GAN for synthetic minority data generation and class balancing
3. Diffusion Model for generating synthetic patient trajectories

The project also provides an opportunity to understand the differences between these architectures, their training procedures, and their practical applications.

## Project Workflow

```text
                    Generative Models
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
         VAE              GAN           Diffusion
          |                |                |
          v                v                v
   ECG-like Signals   Imbalanced Data   Patient Trajectories
          |                |                |
          v                v                v
 Anomaly Detection   Data Augmentation  Synthetic Generation
```

## Technologies Used

| Technology       | Purpose                                  |
| ---------------- | ---------------------------------------- |
| Python           | Core programming language                |
| PyTorch          | Model implementation and training        |
| NumPy            | Numerical operations and data generation |
| Pandas           | Data handling                            |
| Scikit-learn     | Dataset generation and preprocessing     |
| Matplotlib       | Data visualization                       |
| Jupyter Notebook | Development and experimentation          |

## Model 1: Variational Autoencoder

The first part of the project implements a Variational Autoencoder for detecting anomalies in ECG-like signals.

### Dataset

The notebook generates synthetic ECG-like signals containing:

* Periodic signal patterns
* Normal peaks
* Small amounts of noise

It creates:

```text
Normal Samples: 1000
Anomalous Samples: 100
Sequence Length: 100
```

Normal signals contain relatively small noise, while anomalous signals include:

* Higher noise
* Additional abnormal spikes
* Irregular signal behavior

### Data Preprocessing

The generated signals are standardized using `StandardScaler`.

The normalized data is then converted into PyTorch tensors for model training.

### VAE Architecture

The VAE contains:

```text
Input
  |
  v
Encoder
  |
  v
Hidden Layer
  |
  +----------+
  |          |
  v          v
  μ        log variance
  |          |
  +----+-----+
       |
       v
Reparameterization
       |
       v
   Latent Vector
       |
       v
    Decoder
       |
       v
Reconstructed Signal
```

The implemented architecture uses:

```text
Input Dimension: 100
Hidden Dimension: 64
Latent Dimension: 16
```

### Encoder

The encoder maps the input signal into a latent probability distribution.

It produces:

```text
μ
log variance
```

The latent representation is then sampled using the reparameterization trick.

### Reparameterization Trick

The latent vector is generated using:

```text
z = μ + ε × σ
```

where ε is sampled from a standard normal distribution.

This allows the stochastic sampling operation to remain differentiable during training.

### Decoder

The decoder takes the latent representation and reconstructs the original signal.

The goal is to reconstruct normal signals accurately.

### VAE Loss

The VAE uses two loss components:

```text
Total Loss = Reconstruction Loss + KL Divergence
```

Reconstruction loss measures how closely the reconstructed signal matches the original signal.

KL divergence regularizes the latent distribution so that it remains close to a standard normal distribution.

### Anomaly Detection

After training, reconstruction errors are calculated for normal and anomalous samples.

The notebook compares:

```text
Normal Reconstruction Error
Anomalous Reconstruction Error
```

A threshold is calculated using the 95th percentile of the normal reconstruction errors.

Samples whose reconstruction error exceeds this threshold are classified as anomalies.

This demonstrates how VAEs can be used for unsupervised anomaly detection.

## Model 2: Generative Adversarial Network

The second part implements a GAN for synthetic data generation and class balancing.

### Dataset

An imbalanced two-dimensional classification dataset is generated using Scikit-learn.

The dataset contains:

```text
Total Samples: 1000
Class 0: 90%
Class 1: 10%
```

The minority class is extracted and used as the training data for the GAN.

### Objective

The goal is to generate additional synthetic minority-class samples and use them to reduce the class imbalance.

The workflow is:

```text
Imbalanced Dataset
        |
        v
Extract Minority Class
        |
        v
Train GAN
        |
        v
Generate Synthetic Minority Samples
        |
        v
Combine Real + Synthetic Data
        |
        v
Balanced Dataset
```

### GAN Architecture

The GAN consists of two neural networks:

```text
Generator
Discriminator
```

#### Generator

The Generator receives random noise and attempts to produce realistic minority-class samples.

Architecture:

```text
Noise Vector
    |
    v
Linear Layer
    |
    v
ReLU
    |
    v
Linear Layer
    |
    v
ReLU
    |
    v
Generated Sample
```

The noise dimension is:

```text
10
```

The output dimension is:

```text
2
```

because the dataset contains two features.

#### Discriminator

The Discriminator receives a sample and predicts whether it is real or generated.

Architecture:

```text
Input Sample
    |
    v
Linear Layer
    |
    v
LeakyReLU
    |
    v
Linear Layer
    |
    v
LeakyReLU
    |
    v
Linear Layer
    |
    v
Sigmoid
```

### GAN Training

The Generator and Discriminator are trained in an adversarial process.

The Discriminator learns to distinguish:

```text
Real Minority Data
```

from:

```text
Generated Minority Data
```

The Generator learns to create samples that can fool the Discriminator.

The notebook trains the GAN for:

```text
2000 epochs
```

using Binary Cross Entropy loss.

### Synthetic Data Generation

After training, the Generator creates:

```text
900 synthetic minority samples
```

These samples are transformed back to the original feature scale and combined with the original dataset.

The final dataset contains:

```text
Original Data + Synthetic Minority Data
```

The resulting distribution is visualized to compare real and generated minority samples.

## Model 3: Diffusion Model

The third part implements a simplified diffusion model for generating synthetic patient trajectories.

### Dataset

The notebook generates synthetic patient trajectories.

The dataset contains:

```text
Patients: 1000
Trajectory Length: 20
```

Each trajectory contains a combination of:

* Starting patient state
* Gradual trend
* Sinusoidal variation
* Small random noise

### Example Trajectory

The general structure is:

```text
Initial State
     |
     v
Time Step 1
     |
     v
Time Step 2
     |
     v
...
     |
     v
Time Step 20
```

The trajectories are standardized before being converted into PyTorch tensors.

## Forward Diffusion Process

The diffusion model first adds noise to the original data.

The notebook uses:

```text
Total Diffusion Steps: 100
```

A linear beta schedule is created:

```text
β = 0.0001 → 0.02
```

The corresponding alpha values and cumulative alpha products are calculated.

The forward process follows the general form:

```text
xₜ = √α̅ₜ x₀ + √(1 - α̅ₜ) ε
```

where:

* `x₀` is the original data
* `xₜ` is the noisy data at timestep t
* `α̅ₜ` controls the remaining signal
* `ε` is Gaussian noise

The notebook visualizes several diffusion stages:

```text
Step 0
Step 25
Step 50
Step 75
Step 99
```

This demonstrates how the original trajectory becomes progressively noisier.

## Diffusion Denoising Network

A neural network is trained to predict the noise added to the data.

The model uses a time embedding so that it can understand which diffusion timestep it is processing.

Architecture:

```text
Noisy Trajectory
        +
Time Embedding
        |
        v
Concatenation
        |
        v
Hidden Layer
        |
        v
Hidden Layer
        |
        v
Predicted Noise
```

The model uses:

```text
Input Dimension: 20
Hidden Dimension: 128
Time Embedding Dimension: 32
Diffusion Steps: 100
```

## Diffusion Training

The model is trained for:

```text
500 epochs
```

The training objective is to minimize the Mean Squared Error between:

```text
Actual Noise
```

and:

```text
Predicted Noise
```

The training loss is recorded and visualized throughout training.

## Reverse Diffusion

After training, the model generates new trajectories starting from random Gaussian noise.

The reverse process iteratively removes predicted noise:

```text
Random Noise
     |
     v
Step 99
     |
     v
Step 98
     |
     v
...
     |
     v
Step 0
     |
     v
Generated Trajectory
```

The notebook generates synthetic patient trajectories and compares them with the original trajectories.

## Generated Data Analysis

The generated trajectories are evaluated using basic statistical comparisons.

The notebook calculates:

```text
Real Mean
Generated Mean

Real Standard Deviation
Generated Standard Deviation
```

These values provide a basic comparison between the original and generated data distributions.

## Comparison of the Three Models

| Model     | Main Application     | Training Objective             | Output                     |
| --------- | -------------------- | ------------------------------ | -------------------------- |
| VAE       | Anomaly Detection    | Reconstruction + KL Divergence | Reconstructed Signals      |
| GAN       | Data Augmentation    | Adversarial Training           | Synthetic Minority Samples |
| Diffusion | Synthetic Generation | Noise Prediction               | Synthetic Trajectories     |

### VAE

The VAE learns a probabilistic latent representation and reconstructs the input data.

In this project, reconstruction error is used to identify anomalous ECG-like signals.

### GAN

The GAN uses competition between a Generator and Discriminator.

The Generator learns to create synthetic samples that resemble the minority class.

### Diffusion Model

The diffusion model learns to predict noise at different timesteps and uses the learned denoising process to generate new trajectories.

## Model Characteristics

The notebook highlights the following general characteristics:

### VAE

* Uses a probabilistic latent space
* Supports reconstruction
* Useful for anomaly detection
* Provides diverse latent representations
* Can produce smoother outputs

### GAN

* Uses adversarial training
* Generator creates synthetic samples
* Discriminator evaluates generated samples
* Useful for data augmentation
* Training can be sensitive to model balance

### Diffusion Model

* Uses a forward noising process
* Learns a reverse denoising process
* Generates samples through iterative refinement
* Can provide stable generation
* Requires multiple inference steps

## Learning Objectives

This project provides practical experience with:

* Variational Autoencoders
* Encoder and decoder architectures
* Latent representations
* Reparameterization trick
* KL divergence
* Reconstruction loss
* Anomaly detection
* Generative Adversarial Networks
* Generator and Discriminator networks
* Adversarial training
* Synthetic data generation
* Data augmentation
* Diffusion processes
* Noise schedules
* Forward diffusion
* Reverse diffusion
* Time embeddings
* Noise prediction
* Generative modeling with PyTorch

## Installation

Install the required libraries:

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-link>
```

### 2. Navigate to the Project

```bash
cd <repository-name>
```

### 3. Install Dependencies

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

### 4. Open the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
gan-vae-diff.ipynb
```

### 5. Run the Notebook

Run the cells sequentially.

The notebook will:

1. Generate synthetic ECG-like data
2. Train the VAE
3. Calculate reconstruction errors
4. Detect anomalous signals
5. Create an imbalanced classification dataset
6. Train the GAN
7. Generate synthetic minority samples
8. Create an augmented dataset
9. Generate synthetic patient trajectories
10. Train the diffusion model
11. Generate new trajectories
12. Compare real and generated data

## Project Structure

```text
Generative-Models/
|
├── gan-vae-diff.ipynb
└── README.md
```

## Applications

The techniques explored in this project can be applied to several areas.

### VAE

* Anomaly detection
* Representation learning
* Data reconstruction
* Dimensionality reduction

### GAN

* Data augmentation
* Synthetic dataset generation
* Class imbalance handling
* Generative modeling

### Diffusion Models

* Synthetic time-series generation
* Data generation
* Generative AI
* Probabilistic modeling
* Image and signal generation

## Limitations

This notebook is primarily an educational implementation.

The datasets used for the experiments are synthetic rather than real-world medical or production datasets.

The diffusion implementation is simplified and focuses on trajectory generation rather than image generation.

The project does not include:

* Production-scale training
* Large datasets
* Advanced VAE architectures
* Conditional GANs
* Wasserstein GANs
* U-Net diffusion architectures
* Large-scale diffusion training
* Advanced evaluation metrics
* Privacy analysis of generated data

## Future Improvements

Possible extensions include:

* Use real ECG datasets for VAE anomaly detection
* Add ROC-AUC, precision, recall, and F1 evaluation
* Experiment with β-VAE
* Implement conditional GANs
* Experiment with WGAN or WGAN-GP
* Add batch-based training
* Implement more advanced diffusion architectures
* Experiment with different noise schedules
* Add quantitative evaluation of generated trajectories
* Compare multiple generative architectures on the same dataset
* Build an interactive application for generated samples

## Conclusion

This project provides a hands-on comparison of three major generative modeling approaches using PyTorch.

The VAE demonstrates how reconstruction error can be used for anomaly detection. The GAN demonstrates how synthetic samples can be generated to address class imbalance. The simplified Diffusion Model demonstrates how noisy trajectories can be transformed into synthetic data through a learned denoising process.

Together, these experiments provide a practical foundation for understanding how VAEs, GANs, and Diffusion Models approach generative modeling from different perspectives.

## Author

Aayush Kumar

