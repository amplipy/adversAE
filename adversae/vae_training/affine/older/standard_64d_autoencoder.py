"""
Standard 64D Affine-Invariant Autoencoder
This module contains classes and functions specific to the standard 64-dimensional autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import shared components from the shared module
import affine_autoencoder_shared as shared


class AutoEncoder(nn.Module):
    """
    Standard autoencoder for learning digit representations
    without geometric transformation information.
    """
    def __init__(self, latent_dim=64):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 7x7 -> 4x4
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 128 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # 4x4 -> 7x7
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 7x7 -> 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 14x14 -> 28x28
            nn.Sigmoid()
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


class AffineInvariantAutoEncoder(nn.Module):
    """
    Combined model with autoencoder and affine transformation branch
    for disentangling geometric transformations from digit identity.
    """
    def __init__(self, latent_dim=64):
        super(AffineInvariantAutoEncoder, self).__init__()
        
        self.autoencoder = AutoEncoder(latent_dim)
        self.affine_net = shared.AffineTransformationNetwork()
        
    def apply_affine_transformation(self, images, affine_params):
        """
        Apply affine transformation to images using the predicted parameters.
        
        Args:
            images: Input images [batch_size, 1, 28, 28]
            affine_params: Affine parameters [batch_size, 6] -> [a, b, c, d, tx, ty]
        
        Returns:
            Transformed images
        """
        batch_size = images.size(0)
        
        # Reshape affine parameters to transformation matrices
        # [a, b, tx]
        # [c, d, ty]
        theta = affine_params.view(-1, 2, 3)
        
        # Create sampling grid
        grid = F.affine_grid(theta, images.size(), align_corners=False)
        
        # Apply transformation
        transformed_images = F.grid_sample(images, grid, align_corners=False)
        
        return transformed_images
    
    def forward(self, x):
        # Get reconstruction from autoencoder
        reconstruction, latent = self.autoencoder(x)
        
        # Predict affine transformation parameters
        affine_params = self.affine_net(x)
        
        # Apply affine transformation to reconstruction
        transformed_reconstruction = self.apply_affine_transformation(reconstruction, affine_params)
        
        return reconstruction, transformed_reconstruction, latent, affine_params


def train_affine_invariant_autoencoder(model, train_loader, epochs=20, lr=1e-3, alpha=1.0, beta=0.1, gamma=0.01):
    """Train the affine-invariant autoencoder"""
    # Force CPU for affine operations due to MPS grid_sampler limitation
    device = torch.device('cpu')
    print(f"Training on device: {device} (CPU forced for affine grid operations)")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    train_losses = []
    recon_losses = []
    affine_losses = []
    reg_losses = []
    
    print(f"Training on device: {device}")
    print("Training Affine-Invariant Autoencoder...")
    
    for epoch in tqdm(range(epochs)):
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_affine_loss = 0
        epoch_reg_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, transformed_reconstruction, latent, affine_params = model(data)
            
            # Loss calculation
            total_loss, recon_loss, affine_loss, reg_loss = shared.affine_invariant_loss(
                data, reconstruction, transformed_reconstruction, affine_params, alpha, beta, gamma)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_affine_loss += affine_loss.item()
            epoch_reg_loss += reg_loss.item()
        
        # Average losses for the epoch
        avg_total_loss = epoch_total_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_affine_loss = epoch_affine_loss / len(train_loader)
        avg_reg_loss = epoch_reg_loss / len(train_loader)
        
        train_losses.append(avg_total_loss)
        recon_losses.append(avg_recon_loss)
        affine_losses.append(avg_affine_loss)
        reg_losses.append(avg_reg_loss)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch}: Total={avg_total_loss:.6f}, Recon={avg_recon_loss:.6f}, '
                  f'Affine={avg_affine_loss:.6f}, Reg={avg_reg_loss:.6f}')
    
    return {
        'total_losses': train_losses,
        'recon_losses': recon_losses,
        'affine_losses': affine_losses,
        'reg_losses': reg_losses
    }


def visualize_standard_autoencoder_results(model, test_loader, device, num_samples=8):
    """Visualize the results of standard 64D affine-invariant autoencoder"""
    model.eval()
    
    # Get test samples
    test_iter = iter(test_loader)
    test_images, test_labels = next(test_iter)
    test_images = test_images[:num_samples].to(device)
    test_labels = test_labels[:num_samples]
    
    with torch.no_grad():
        reconstruction, transformed_reconstruction, latent, affine_params = model(test_images)
    
    # Visualize results
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 2, 8))
    
    for i in range(num_samples):
        # Original image
        axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original\nLabel: {test_labels[i].item()}')
        axes[0, i].axis('off')
        
        # Direct reconstruction
        axes[1, i].imshow(reconstruction[i].cpu().squeeze(), cmap='gray')
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
        
        # Affine-transformed reconstruction
        axes[2, i].imshow(transformed_reconstruction[i].cpu().squeeze(), cmap='gray')
        axes[2, i].set_title('Affine-Transformed\nReconstruction')
        axes[2, i].axis('off')
        
        # Affine parameters visualization
        params = affine_params[i].cpu().numpy()
        axes[3, i].text(0.1, 0.8, f'a={params[0]:.2f} b={params[1]:.2f}', transform=axes[3, i].transAxes)
        axes[3, i].text(0.1, 0.6, f'c={params[2]:.2f} d={params[3]:.2f}', transform=axes[3, i].transAxes)
        axes[3, i].text(0.1, 0.4, f'tx={params[4]:.2f} ty={params[5]:.2f}', transform=axes[3, i].transAxes)
        axes[3, i].set_title('Affine Params')
        axes[3, i].axis('off')
    
    plt.suptitle('Standard 64D Affine-Invariant Autoencoder Results', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Print affine parameters statistics
    print("\nAffine Parameters Statistics:")
    print(f"Mean parameters: {affine_params.mean(dim=0).cpu().numpy()}")
    print(f"Std parameters: {affine_params.std(dim=0).cpu().numpy()}")
    
    # Print latent space statistics
    print(f"\nLatent Space Statistics (64D):")
    print(f"Latent shape: {latent.shape}")
    print(f"Latent mean: {latent.mean(dim=0).cpu().numpy()[:5]}... (showing first 5)")
    print(f"Latent std: {latent.std(dim=0).cpu().numpy()[:5]}... (showing first 5)")


def visualize_64d_latent_space(model, test_loader, device, num_samples=1000):
    """Visualize standard 64D latent space (first 2 dimensions)"""
    model.eval()
    
    latents = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if len(latents) * test_loader.batch_size >= num_samples:
                break
                
            data = data.to(device)
            _, _, latent, _ = model(data)
            
            latents.append(latent.cpu().numpy())
            labels.append(label.numpy())
    
    latent_data = np.concatenate(latents, axis=0)[:num_samples]
    label_data = np.concatenate(labels, axis=0)[:num_samples]
    
    # Plot first 2 dimensions
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_data[:, 0], latent_data[:, 1], c=label_data, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Standard 64D Autoencoder - Latent Space Visualization (First 2 Dims)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"📊 64D Latent Space Analysis:")
    print(f"  Latent space shape: {latent_data.shape}")
    print(f"  Dimensions visualized: 2 out of {latent_data.shape[1]}")
    print(f"  Latent range: X[{latent_data[:, 0].min():.2f}, {latent_data[:, 0].max():.2f}], Y[{latent_data[:, 1].min():.2f}, {latent_data[:, 1].max():.2f}]")
    
    return latent_data, label_data


def scan_64d_latent_grid(model, device, grid_size=10, latent_range=(-2, 2)):
    """Scan through 64D latent space grid (first 2 dimensions) and show decoder output"""
    model.eval()
    
    # Create a grid of latent space coordinates
    x = np.linspace(latent_range[0], latent_range[1], grid_size)
    y = np.linspace(latent_range[0], latent_range[1], grid_size)
    
    # Initialize the latent vectors (64D, all other dimensions set to 0)
    latent_vectors = torch.zeros(grid_size * grid_size, 64, device=device)
    
    # Fill in the first two dimensions with our grid coordinates
    for i, y_val in enumerate(y):
        for j, x_val in enumerate(x):
            idx = i * grid_size + j
            latent_vectors[idx, 0] = x_val  # First latent dimension
            latent_vectors[idx, 1] = y_val  # Second latent dimension
    
    # Generate images from latent vectors
    with torch.no_grad():
        # Get raw decoder output
        raw_images = model.autoencoder.decoder(latent_vectors)
    
    # Create visualization
    images = raw_images.cpu().numpy()
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 1.2, grid_size * 1.2))
    
    if grid_size == 1:
        axes = [axes]  # Handle single subplot case
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            img = images[idx].squeeze()
            
            if grid_size == 1:
                ax = axes[0]
            else:
                ax = axes[i, j] if grid_size > 1 else axes[j]
                
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            # Add coordinates for corner and center images
            if (i == 0 and j == 0) or (i == grid_size//2 and j == grid_size//2) or (i == grid_size-1 and j == grid_size-1):
                ax.set_title(f'({x[j]:.1f},{y[i]:.1f})', fontsize=8)
    
    # Add overall title and axis labels
    plt.suptitle('64D Latent Space Grid Scan (Raw Decoder Output)\nFirst 2 Dimensions', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 64D Latent Space Grid Analysis:")
    print(f"  Grid size: {grid_size} × {grid_size} = {grid_size*grid_size} generated images")
    print(f"  Latent dimensions scanned: 2 out of 64")
    print(f"  Scan range: {latent_range[0]} to {latent_range[1]} for both dimensions")
    print(f"  Other 62 latent dimensions: set to 0")
    
    return images, latent_vectors.cpu().numpy()
