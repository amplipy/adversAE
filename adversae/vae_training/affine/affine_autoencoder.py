"""
Affine-Invariant Autoencoder for MNIST
This implementation creates autoencoders with affine transformation branches
to disentangle geometric transformations from digit identity.

Includes:
- Classic 64D Autoencoder
- Structured 2D+6D Autoencoder  
- Cloud-optimized training functions
- Comprehensive visualization tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import os
import json
from datetime import datetime


class AffineTransformationNetwork(nn.Module):
    """
    Parallel network branch that predicts affine transformation parameters
    from the input image to make the autoencoder invariant to geometric transformations.
    """
    def __init__(self):
        super(AffineTransformationNetwork, self).__init__()
        
        # Convolutional feature extractor for affine parameters
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  # 14x14 -> 14x14
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 7x7 -> 7x7
        self.pool3 = nn.MaxPool2d(2, 2)  # 7x7 -> 3x3 (with some rounding)
        
        # Fully connected layers to predict affine parameters
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Output 6 affine transformation parameters [a, b, c, d, tx, ty]
        # for transformation matrix: [[a, b, tx], [c, d, ty], [0, 0, 1]]
        self.fc_affine = nn.Linear(64, 6)
        
        # Initialize affine parameters to identity transformation
        self.fc_affine.weight.data.zero_()
        self.fc_affine.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def forward(self, x):
        # Extract features for affine parameter prediction
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten and predict affine parameters
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output affine parameters
        affine_params = self.fc_affine(x)
        
        return affine_params


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
        self.affine_net = AffineTransformationNetwork()
        
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


def affine_invariant_loss(original, reconstruction, transformed_reconstruction, affine_params, 
                         alpha=1.0, beta=0.1, gamma=0.01):
    """
    Combined loss function for affine-invariant autoencoder.
    
    Args:
        original: Original input images
        reconstruction: Direct autoencoder reconstruction
        transformed_reconstruction: Affine-transformed reconstruction
        affine_params: Predicted affine transformation parameters
        alpha: Weight for reconstruction loss
        beta: Weight for affine-invariant loss
        gamma: Weight for regularization loss
    
    Returns:
        Total loss, reconstruction loss, affine loss, regularization loss
    """
    # Standard reconstruction loss
    recon_loss = F.mse_loss(reconstruction, original)
    
    # Affine-invariant loss (transformed reconstruction should match original)
    affine_loss = F.mse_loss(transformed_reconstruction, original)
    
    # Regularization loss to prevent extreme transformations
    # Encourage affine parameters to stay close to identity transformation
    identity_params = torch.tensor([1, 0, 0, 0, 1, 0], device=affine_params.device, dtype=torch.float32).unsqueeze(0)
    identity_params = identity_params.expand_as(affine_params)
    reg_loss = F.mse_loss(affine_params, identity_params)
    
    # Combined loss
    total_loss = alpha * recon_loss + beta * affine_loss + gamma * reg_loss
    
    return total_loss, recon_loss, affine_loss, reg_loss


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
            
            # Calculate losses
            total_loss, recon_loss, affine_loss, reg_loss = affine_invariant_loss(
                data, reconstruction, transformed_reconstruction, affine_params, alpha, beta, gamma
            )
            
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


def visualize_affine_results(model, test_loader, device, num_samples=8):
    """Visualize the results of affine-invariant autoencoder"""
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
    
    plt.tight_layout()
    plt.show()
    
    # Print affine parameters statistics
    print("\nAffine Parameters Statistics:")
    print(f"Mean parameters: {affine_params.mean(dim=0).cpu().numpy()}")
    print(f"Std parameters: {affine_params.std(dim=0).cpu().numpy()}")


def analyze_latent_disentanglement(model, test_loader, device):
    """Analyze how well the model disentangles content from transformation"""
    model.eval()
    
    latents = []
    labels = []
    affine_params_list = []
    
    # Collect latent representations and affine parameters
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if batch_idx >= 20:  # Limit samples for analysis
                break
                
            data = data.to(device)
            _, _, latent, affine_params = model(data)
            
            latents.append(latent.cpu().numpy())
            labels.append(label.numpy())
            affine_params_list.append(affine_params.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    affine_params_array = np.concatenate(affine_params_list, axis=0)
    
    # Visualize latent space (first 2 dimensions)
    plt.figure(figsize=(15, 5))
    
    # Plot latent space colored by digit
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space (by Digit)')
    
    # Plot affine parameter space
    plt.subplot(1, 3, 2)
    plt.scatter(affine_params_array[:, 0], affine_params_array[:, 3], c=labels, cmap='tab10', alpha=0.6)
    plt.xlabel('Affine Parameter a')
    plt.ylabel('Affine Parameter d')
    plt.title('Affine Space (a vs d)')
    
    # Plot translation parameters
    plt.subplot(1, 3, 3)
    plt.scatter(affine_params_array[:, 4], affine_params_array[:, 5], c=labels, cmap='tab10', alpha=0.6)
    plt.xlabel('Translation tx')
    plt.ylabel('Translation ty')
    plt.title('Translation Parameters')
    
    plt.tight_layout()
    plt.show()
    
    return latents, affine_params_array


def visualize_latent_space_with_reconstructions(model, test_loader, device, num_samples=100):
    """
    Visualize the latent space with actual reconstruction images overlaid on the 2D projection.
    This shows the direct autoencoder reconstructions (before affine transformation).
    """
    model.eval()
    
    latents = []
    labels = []
    reconstructions = []
    
    # Collect latent representations and reconstructions
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if len(latents) * test_loader.batch_size >= num_samples:
                break
                
            data = data.to(device)
            reconstruction, _, latent, _ = model(data)
            
            latents.append(latent.cpu().numpy())
            labels.append(label.numpy())
            reconstructions.append(reconstruction.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    reconstructions = np.concatenate(reconstructions, axis=0)[:num_samples]
    
    # Create figure with latent space visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: Scatter plot colored by digit
    scatter = ax1.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.7, s=50)
    ax1.set_xlabel('Latent Dimension 1', fontsize=12)
    ax1.set_ylabel('Latent Dimension 2', fontsize=12)
    ax1.set_title('Latent Space Colored by Digit Class', fontsize=14)
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Digit Class', fontsize=12)
    
    # Right plot: Images overlaid on latent space
    ax2.set_xlim(latents[:, 0].min() - 1, latents[:, 0].max() + 1)
    ax2.set_ylim(latents[:, 1].min() - 1, latents[:, 1].max() + 1)
    ax2.set_xlabel('Latent Dimension 1', fontsize=12)
    ax2.set_ylabel('Latent Dimension 2', fontsize=12)
    ax2.set_title('Latent Space with Autoencoder Reconstructions', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Sample every nth point to avoid overcrowding
    step = max(1, num_samples // 50)  # Show at most 50 images
    
    for i in range(0, num_samples, step):
        # Get the reconstruction image
        img = reconstructions[i].squeeze()
        
        # Position and size for the image
        x, y = latents[i, 0], latents[i, 1]
        
        # Create a small inset axes for the image
        img_size = 0.8  # Size of the image in data coordinates
        extent = [x - img_size/2, x + img_size/2, y - img_size/2, y + img_size/2]
        
        # Show the reconstruction image
        ax2.imshow(img, extent=extent, cmap='gray', alpha=0.8, aspect='equal')
        
        # Add a colored border based on the digit class
        color = plt.cm.tab10(labels[i] / 9.0)
        rect = plt.Rectangle((x - img_size/2, y - img_size/2), img_size, img_size, 
                           linewidth=2, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n📊 Latent Space Analysis:")
    print(f"  Total samples: {num_samples}")
    print(f"  Latent dimensions: {latents.shape[1]}")
    print(f"  Visualized images: {len(range(0, num_samples, step))}")
    print(f"  Latent space range: X[{latents[:, 0].min():.2f}, {latents[:, 0].max():.2f}], Y[{latents[:, 1].min():.2f}, {latents[:, 1].max():.2f}]")
    
    # Analyze clustering by computing within-class vs between-class distances
    print(f"\n🔍 Clustering Quality Analysis:")
    for digit in range(10):
        digit_mask = labels == digit
        if np.sum(digit_mask) > 1:
            digit_latents = latents[digit_mask]
            # Within-class variance (compactness)
            within_var = np.var(digit_latents, axis=0).mean()
            print(f"  Digit {digit}: within-class variance = {within_var:.4f} (lower = better clustering)")
    
    return latents, labels, reconstructions


def visualize_latent_space_grid(model, device, latent_dim=64, grid_size=15, latent_range=(-3, 3), show_corrected=True):
    """
    Create a grid visualization by scanning across the first two latent dimensions.
    Shows both raw decoder output and affine-corrected clean digits.
    
    Args:
        model: Trained AffineInvariantAutoEncoder
        device: Device to run inference on
        latent_dim: Dimensionality of the latent space
        grid_size: Number of points along each axis (grid_size x grid_size total)
        latent_range: Range of values to scan for each latent dimension
        show_corrected: If True, shows side-by-side raw and corrected outputs
    """
    model.eval()
    
    # Create a grid of latent space coordinates
    x = np.linspace(latent_range[0], latent_range[1], grid_size)
    y = np.linspace(latent_range[0], latent_range[1], grid_size)
    
    # Initialize the latent vectors (all other dimensions set to 0)
    latent_vectors = torch.zeros(grid_size * grid_size, latent_dim, device=device)
    
    # Fill in the first two dimensions with our grid coordinates
    for i, y_val in enumerate(y):
        for j, x_val in enumerate(x):
            idx = i * grid_size + j
            latent_vectors[idx, 0] = x_val  # First latent dimension
            latent_vectors[idx, 1] = y_val  # Second latent dimension
    
    # Generate images from latent vectors
    with torch.no_grad():
        # Get raw decoder output (potentially transformed/gibberish)
        raw_images = model.autoencoder.decoder(latent_vectors)
        
        if show_corrected:
            # To get corrected images, we need to pass through the full model
            # Use the raw images as input to get affine parameters and corrected outputs
            
            # Get affine parameters by passing raw images through affine network
            affine_params = model.affine_net(raw_images.unsqueeze(1) if raw_images.dim() == 3 else raw_images)
            
            # Apply affine transformation to get clean digits
            corrected_images = model.apply_affine_transformation(raw_images, affine_params)
            
            # Convert to numpy
            raw_images_np = raw_images.cpu().numpy()
            corrected_images_np = corrected_images.cpu().numpy()
            
            # Create side-by-side visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(grid_size * 2, grid_size * 1))
            
            # Create grid for raw images (left)
            raw_grid = np.zeros((grid_size * 28, grid_size * 28))
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    img = raw_images_np[idx].squeeze()
                    y_start, y_end = i * 28, (i + 1) * 28
                    x_start, x_end = j * 28, (j + 1) * 28
                    raw_grid[y_start:y_end, x_start:x_end] = img
            
            # Create grid for corrected images (right)
            corrected_grid = np.zeros((grid_size * 28, grid_size * 28))
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    img = corrected_images_np[idx].squeeze()
                    y_start, y_end = i * 28, (i + 1) * 28
                    x_start, x_end = j * 28, (j + 1) * 28
                    corrected_grid[y_start:y_end, x_start:x_end] = img
            
            # Display the grids
            ax1.imshow(raw_grid, cmap='gray', vmin=0, vmax=1)
            ax1.set_title('Raw Decoder Output\n(Potentially Transformed/Gibberish)', fontsize=14)
            ax1.axis('off')
            
            ax2.imshow(corrected_grid, cmap='gray', vmin=0, vmax=1)
            ax2.set_title('Affine-Corrected Clean Digits\n(What You Expected!)', fontsize=14)
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\n✅ Grid visualization complete!")
            print(f"📊 Generated {grid_size}x{grid_size} = {grid_size*grid_size} image pairs")
            print(f"🎯 Key Insight: The 'gibberish' on the left becomes clean digits on the right!")
            
            # Return both versions
            return raw_images_np, corrected_images_np, latent_vectors.cpu().numpy()
            
        else:
            # Show only raw decoder output (original behavior)
            images = raw_images.cpu().numpy()
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 1.5, grid_size * 1.5))
            
            if grid_size == 1:
                axes = [axes]  # Handle single subplot case
            
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    img = images[idx].squeeze()
                    
                    if grid_size == 1:
                        ax = axes[0]
                    else:
                        ax = axes[i, j] if grid_size > 1 else axes[idx]
                    
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    
                    # Add coordinates as title for corner and center images
                    if (i == 0 and j == 0) or (i == grid_size//2 and j == grid_size//2) or (i == grid_size-1 and j == grid_size-1):
                        x_coord = latent_vectors[idx, 0].item()
                        y_coord = latent_vectors[idx, 1].item()
                        ax.set_title(f'({x_coord:.1f}, {y_coord:.1f})', fontsize=8)
            
            # Add overall title and axis labels
            fig.suptitle('Latent Space Grid Scan (Raw Decoder Output)', fontsize=16, y=0.98)
            
            # Add axis labels
            fig.text(0.5, 0.02, f'Latent Dimension 1 (range: {latent_range[0]} to {latent_range[1]})', 
                     ha='center', fontsize=12)
            fig.text(0.02, 0.5, f'Latent Dimension 2 (range: {latent_range[0]} to {latent_range[1]})', 
                     va='center', rotation=90, fontsize=12)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.93)
            plt.show()
            
            return images, latent_vectors.cpu().numpy()
    
    # Print analysis
    print(f"\n📊 Latent Space Grid Analysis:")
    print(f"  Grid size: {grid_size} × {grid_size} = {grid_size*grid_size} generated images")
    print(f"  Latent dimensions scanned: 2 out of {latent_dim}")
    print(f"  Scan range: {latent_range[0]} to {latent_range[1]} for both dimensions")
    print(f"  Other latent dimensions: set to 0")
    
    # Analyze the generated images
    image_stats = {
        'mean_pixel_value': np.mean(images),
        'std_pixel_value': np.std(images),
        'min_pixel_value': np.min(images),
        'max_pixel_value': np.max(images)
    }
    
    print(f"\n🎨 Generated Image Statistics:")
    print(f"  Mean pixel value: {image_stats['mean_pixel_value']:.4f}")
    print(f"  Std pixel value: {image_stats['std_pixel_value']:.4f}")
    print(f"  Pixel range: [{image_stats['min_pixel_value']:.4f}, {image_stats['max_pixel_value']:.4f}]")
    
    # Count how many images look like recognizable digits
    print(f"\n🔍 Visual Quality Assessment:")
    print("  Look for:")
    print("  • Smooth transitions between neighboring grid points")
    print("  • Recognizable digit-like structures in various regions")
    print("  • Clear boundaries between different digit types")
    print("  • Gradual morphing from one digit type to another")
    
    return images, latent_vectors.cpu().numpy()


def visualize_latent_space_interpolation(model, test_loader, device, num_interpolations=10):
    """
    Create interpolations between real digit latent representations
    to show smooth transitions in the decoder output (before affine transformations).
    """
    model.eval()
    
    # Get some real test samples to interpolate between
    test_iter = iter(test_loader)
    test_images, test_labels = next(test_iter)
    
    # Select two different digits for interpolation
    digit_indices = {}
    for i, label in enumerate(test_labels):
        if label.item() not in digit_indices:
            digit_indices[label.item()] = i
        if len(digit_indices) >= 4:  # Get at least 4 different digits
            break
    
    # Create interpolations between pairs of different digits
    interpolation_pairs = [
        (list(digit_indices.keys())[0], list(digit_indices.keys())[1]),
        (list(digit_indices.keys())[2], list(digit_indices.keys())[3]) if len(digit_indices) >= 4 else (list(digit_indices.keys())[0], list(digit_indices.keys())[2])
    ]
    
    fig, axes = plt.subplots(len(interpolation_pairs), num_interpolations + 2, 
                            figsize=((num_interpolations + 2) * 1.5, len(interpolation_pairs) * 1.5))
    
    if len(interpolation_pairs) == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for pair_idx, (digit1, digit2) in enumerate(interpolation_pairs):
            # Get latent representations of the two digits
            img1 = test_images[digit_indices[digit1]:digit_indices[digit1]+1].to(device)
            img2 = test_images[digit_indices[digit2]:digit_indices[digit2]+1].to(device)
            
            _, _, latent1, _ = model(img1)
            _, _, latent2, _ = model(img2)
            
            # Show original images
            axes[pair_idx, 0].imshow(img1.cpu().squeeze(), cmap='gray')
            axes[pair_idx, 0].set_title(f'Digit {digit1}')
            axes[pair_idx, 0].axis('off')
            
            axes[pair_idx, -1].imshow(img2.cpu().squeeze(), cmap='gray')
            axes[pair_idx, -1].set_title(f'Digit {digit2}')
            axes[pair_idx, -1].axis('off')
            
            # Create interpolations
            for i in range(num_interpolations):
                alpha = i / (num_interpolations - 1)
                interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
                
                # Generate image using only the decoder (before affine transformation)
                generated_img = model.autoencoder.decoder(interpolated_latent)
                
                axes[pair_idx, i + 1].imshow(generated_img.cpu().squeeze(), cmap='gray')
                axes[pair_idx, i + 1].set_title(f'α={alpha:.2f}')
                axes[pair_idx, i + 1].axis('off')
    
    plt.suptitle('Latent Space Interpolations (Decoder Output Before Affine)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔄 Latent Space Interpolation Analysis:")
    print(f"  Interpolation pairs: {len(interpolation_pairs)}")
    print(f"  Steps per interpolation: {num_interpolations}")
    print(f"  Shows smooth transitions between different digit types")
    print(f"  Demonstrates the continuity of the learned latent space")


def plot_training_progress(losses_dict):
    """Plot training loss progression"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Total loss
    axes[0, 0].plot(losses_dict['total_losses'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(losses_dict['recon_losses'], 'r-', linewidth=2)
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Affine loss
    axes[1, 0].plot(losses_dict['affine_losses'], 'g-', linewidth=2)
    axes[1, 0].set_title('Affine-Invariant Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Regularization loss
    axes[1, 1].plot(losses_dict['reg_losses'], 'm-', linewidth=2)
    axes[1, 1].set_title('Regularization Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def get_mnist_loaders(batch_size_train=128, batch_size_test=64, data_dir='./data'):
    """Load MNIST dataset and return train and test data loaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    return train_loader, test_loader


def get_device(force_cpu_for_affine=False):
    """Get the appropriate device (GPU/MPS if available, else CPU)"""
    if force_cpu_for_affine:
        device = torch.device('cpu')
        print(f"Using device: cpu (forced for affine grid operations)")
    elif torch.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: mps")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: cuda")
    else:
        device = torch.device('cpu')
        print(f"Using device: cpu")
    return device


def save_model(model, filepath):
    """Save model state dict"""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device):
    """Load model state dict"""
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"Model loaded from {filepath}")
    return model


def demonstrate_transformation_invariance(model, test_loader, device):
    """
    Demonstrate that the autoencoder learns transformation-invariant representations
    by applying known transformations to test images.
    """
    import torchvision.transforms.functional as TF
    
    model.eval()
    
    # Get a test image
    test_iter = iter(test_loader)
    test_images, test_labels = next(test_iter)
    original_image = test_images[0:1].to(device)
    
    # Apply various transformations
    transformations = {
        'Original': original_image,
        'Rotated 15°': TF.rotate(original_image, 15),
        'Rotated -15°': TF.rotate(original_image, -15),
        'Translated': TF.affine(original_image, angle=0, translate=[3, 3], scale=1, shear=0),
        'Sheared': TF.affine(original_image, angle=0, translate=[0, 0], scale=1, shear=10),
    }
    
    # Get representations for all transformations
    results = {}
    with torch.no_grad():
        for name, img in transformations.items():
            recon, trans_recon, latent, affine_params = model(img)
            results[name] = {
                'image': img,
                'reconstruction': recon,
                'transformed_reconstruction': trans_recon,
                'latent': latent,
                'affine_params': affine_params
            }
    
    # Visualize results
    fig, axes = plt.subplots(len(transformations), 4, figsize=(16, len(transformations) * 3))
    
    for i, (name, result) in enumerate(results.items()):
        # Original transformed image
        axes[i, 0].imshow(result['image'].cpu().squeeze(), cmap='gray')
        axes[i, 0].set_title(f'{name}\nInput')
        axes[i, 0].axis('off')
        
        # Direct reconstruction
        axes[i, 1].imshow(result['reconstruction'].cpu().squeeze(), cmap='gray')
        axes[i, 1].set_title('Direct\nReconstruction')
        axes[i, 1].axis('off')
        
        # Affine-corrected reconstruction
        axes[i, 2].imshow(result['transformed_reconstruction'].cpu().squeeze(), cmap='gray')
        axes[i, 2].set_title('Affine-Corrected\nReconstruction')
        axes[i, 2].axis('off')
        
        # Affine parameters
        params = result['affine_params'].cpu().squeeze().numpy()
        axes[i, 3].text(0.1, 0.8, f'a={params[0]:.2f} b={params[1]:.2f}', transform=axes[i, 3].transAxes)
        axes[i, 3].text(0.1, 0.6, f'c={params[2]:.2f} d={params[3]:.2f}', transform=axes[i, 3].transAxes)
        axes[i, 3].text(0.1, 0.4, f'tx={params[4]:.2f} ty={params[5]:.2f}', transform=axes[i, 3].transAxes)
        axes[i, 3].set_title('Predicted\nAffine Params')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze latent space consistency
    print("\nLatent Space Consistency Analysis:")
    original_latent = results['Original']['latent']
    for name, result in results.items():
        if name != 'Original':
            latent_diff = F.mse_loss(result['latent'], original_latent).item()
            print(f"{name}: Latent MSE difference = {latent_diff:.6f}")


if __name__ == "__main__":
    # Quick test of the implementation
    device = get_device()
    
    # Load data
    train_loader, test_loader = get_mnist_loaders()
    
    # Initialize model
    model = AffineInvariantAutoEncoder(latent_dim=32)
    
    # Train model
    losses = train_affine_invariant_autoencoder(model, train_loader, epochs=5)
    
    # Visualize results
    visualize_affine_results(model, test_loader, device)
    
    print("Affine-Invariant Autoencoder test completed!")
# ADDING CLOUD FUNCTIONS
# =============================================================================
# CLOUD OPTIMIZATION FUNCTIONS 
# =============================================================================

def get_cloud_device(config):
    """Get optimal device for cloud training with configuration options"""
    if config.get('force_cuda', False) and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Cloud CUDA device: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Apple MPS device")
    else:
        device = torch.device('cpu')
        print("💻 CPU device")
    
    return device

class StructuredAutoEncoder(nn.Module):
    """Structured autoencoder with separated content and transform latent spaces"""
    def __init__(self, content_dim=2, transform_dim=6):
        super(StructuredAutoEncoder, self).__init__()
        self.content_dim = content_dim
        self.transform_dim = transform_dim
        
        # Shared encoder backbone
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 7x7 -> 4x4
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(True)
        )
        
        # Content encoder (digit identity)
        self.content_encoder = nn.Linear(512, content_dim)
        
        # Transform encoder (spatial transformations)
        self.transform_encoder = nn.Linear(512, transform_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(content_dim + transform_dim, 512),
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
        # Shared encoding
        shared_features = self.shared_encoder(x)
        
        # Separate content and transform encodings
        content_latent = self.content_encoder(shared_features)
        transform_latent = self.transform_encoder(shared_features)
        
        # Combine and decode
        combined_latent = torch.cat([content_latent, transform_latent], dim=1)
        reconstruction = self.decoder(combined_latent)
        
        return reconstruction, content_latent, transform_latent


class StructuredAffineInvariantAutoEncoder(nn.Module):
    """Combined structured autoencoder with affine transformation network"""
    def __init__(self, content_dim=2, transform_dim=6):
        super(StructuredAffineInvariantAutoEncoder, self).__init__()
        
        self.structured_autoencoder = StructuredAutoEncoder(content_dim, transform_dim)
        self.affine_net = AffineTransformationNetwork()
    
    def apply_affine_transformation(self, images, affine_params):
        """Apply affine transformation to images using the predicted parameters"""
        batch_size = images.size(0)
        
        # Reshape affine parameters to transformation matrices
        theta = affine_params.view(-1, 2, 3)
        
        # Create sampling grid
        grid = F.affine_grid(theta, images.size(), align_corners=False)
        
        # Apply transformation
        transformed_images = F.grid_sample(images, grid, align_corners=False)
        
        return transformed_images
    
    def forward(self, x):
        # Get structured reconstruction
        reconstruction, content_latent, transform_latent = self.structured_autoencoder(x)
        
        # Predict affine transformation parameters
        affine_params = self.affine_net(x)
        
        return reconstruction, content_latent, transform_latent, affine_params

def get_cloud_mnist_loaders(batch_size_train=256, batch_size_test=128, data_dir='../data', pin_memory=True, num_workers=4):
    """Get MNIST data loaders optimized for cloud training"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download to specified directory
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, 
                             pin_memory=pin_memory, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,
                           pin_memory=pin_memory, num_workers=num_workers)
    
    print(f"📊 Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    return train_loader, test_loader

def train_structured_autoencoder_cloud(model, train_loader, test_loader, device, config, scaler=None):
    """Cloud-optimized training for structured autoencoder"""
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                          weight_decay=config.get('weight_decay', 1e-5))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7) if config.get('lr_scheduler', False) else None
    
    losses = {'train_loss': [], 'test_loss': [], 'recon_loss': [], 'affine_loss': [], 
             'content_reg': [], 'transform_reg': []}
    
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        recon_total = 0
        affine_total = 0
        content_reg_total = 0
        transform_reg_total = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data = data.to(device)
            
            with torch.cuda.amp.autocast() if scaler else torch.enable_grad():
                # Forward pass
                reconstructed, content_latent, transform_latent, affine_params = model(data)
                
                # Loss components
                recon_loss = F.mse_loss(reconstructed, data)
                
                # Affine loss - penalize deviation from identity transform
                identity_params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
                                             device=device).expand_as(affine_params)
                affine_loss = F.mse_loss(affine_params, identity_params)
                
                # Regularization losses
                content_reg = torch.mean(content_latent ** 2)
                transform_reg = torch.mean(transform_latent ** 2)
                
                # Combined loss
                total_loss = (config['alpha'] * recon_loss + 
                            config['beta'] * affine_loss +
                            config['gamma'] * content_reg +
                            config['delta'] * transform_reg)
            
            # Backward pass
            optimizer.zero_grad()
            if scaler:
                scaler.scale(total_loss).backward()
                if config.get('gradient_clip', 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                optimizer.step()
            
            train_loss += total_loss.item()
            recon_total += recon_loss.item()
            affine_total += affine_loss.item()
            content_reg_total += content_reg.item()
            transform_reg_total += transform_reg.item()
        
        # Testing
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                reconstructed, content_latent, transform_latent, affine_params = model(data)
                recon_loss = F.mse_loss(reconstructed, data)
                identity_params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
                                             device=device).expand_as(affine_params)
                affine_loss = F.mse_loss(affine_params, identity_params)
                content_reg = torch.mean(content_latent ** 2)
                transform_reg = torch.mean(transform_latent ** 2)
                
                total_loss = (config['alpha'] * recon_loss + 
                            config['beta'] * affine_loss +
                            config['gamma'] * content_reg +
                            config['delta'] * transform_reg)
                test_loss += total_loss.item()
        
        # Record losses
        avg_train = train_loss / len(train_loader)
        avg_test = test_loss / len(test_loader)
        losses['train_loss'].append(avg_train)
        losses['test_loss'].append(avg_test)
        losses['recon_loss'].append(recon_total / len(train_loader))
        losses['affine_loss'].append(affine_total / len(train_loader))
        losses['content_reg'].append(content_reg_total / len(train_loader))
        losses['transform_reg'].append(transform_reg_total / len(train_loader))
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(avg_test)
        
        # Early stopping
        if config.get('early_stopping', False):
            if avg_test < best_test_loss:
                best_test_loss = avg_test
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.get('patience', 10):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Epoch {epoch+1}: Train={avg_train:.6f}, Test={avg_test:.6f}")
    
    return losses

def train_autoencoder_cloud(model, train_loader, test_loader, device, config, scaler=None):
    """Cloud-optimized training for standard autoencoder"""
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                          weight_decay=config.get('weight_decay', 1e-5))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7) if config.get('lr_scheduler', False) else None
    
    losses = {'train_loss': [], 'test_loss': [], 'recon_loss': [], 'affine_loss': []}
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        recon_total = 0
        affine_total = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data = data.to(device)
            
            with torch.cuda.amp.autocast() if scaler else torch.enable_grad():
                reconstructed, latent, affine_params = model(data)
                
                recon_loss = F.mse_loss(reconstructed, data)
                identity_params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
                                             device=device).expand_as(affine_params)
                affine_loss = F.mse_loss(affine_params, identity_params)
                
                total_loss = config['alpha'] * recon_loss + config['beta'] * affine_loss
            
            optimizer.zero_grad()
            if scaler:
                scaler.scale(total_loss).backward()
                if config.get('gradient_clip', 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                optimizer.step()
            
            train_loss += total_loss.item()
            recon_total += recon_loss.item()
            affine_total += affine_loss.item()
        
        # Testing
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                reconstructed, latent, affine_params = model(data)
                recon_loss = F.mse_loss(reconstructed, data)
                identity_params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
                                             device=device).expand_as(affine_params)
                affine_loss = F.mse_loss(affine_params, identity_params)
                total_loss = config['alpha'] * recon_loss + config['beta'] * affine_loss
                test_loss += total_loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_test = test_loss / len(test_loader)
        losses['train_loss'].append(avg_train)
        losses['test_loss'].append(avg_test)
        losses['recon_loss'].append(recon_total / len(train_loader))
        losses['affine_loss'].append(affine_total / len(train_loader))
        
        if scheduler:
            scheduler.step(avg_test)
        
        if config.get('early_stopping', False):
            if avg_test < best_test_loss:
                best_test_loss = avg_test
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.get('patience', 10):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Epoch {epoch+1}: Train={avg_train:.6f}, Test={avg_test:.6f}")
    
    return losses

def save_structured_model_cloud(model, config, losses, content_data, transform_data, device):
    """Save structured model with metadata for cloud deployment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = f"structured_model_{timestamp}.pth"
    metadata_file = f"structured_metadata_{timestamp}.json"
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'device': str(device)
    }, model_file)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'config': config,
        'final_losses': {k: v[-1] if v else 0 for k, v in losses.items()},
        'content_latent_stats': {
            'mean': content_data.mean(axis=0).tolist(),
            'std': content_data.std(axis=0).tolist()
        },
        'transform_latent_stats': {
            'mean': transform_data.mean(axis=0).tolist(),
            'std': transform_data.std(axis=0).tolist()
        }
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved: {model_file}, {metadata_file}")
    return model_file, metadata_file

def save_model_cloud(model, config, losses, device):
    """Save standard model for cloud deployment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = f"autoencoder_model_{timestamp}.pth"
    metadata_file = f"autoencoder_metadata_{timestamp}.json"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'device': str(device)
    }, model_file)
    
    metadata = {
        'timestamp': timestamp,
        'config': config,
        'final_losses': {k: v[-1] if v else 0 for k, v in losses.items()}
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved: {model_file}, {metadata_file}")
    return model_file, metadata_file

def plot_structured_training_progress(losses):
    """Plot training progress for structured autoencoder"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0,0].plot(losses['train_loss'], label='Train', color='blue')
    axes[0,0].plot(losses['test_loss'], label='Test', color='red')
    axes[0,0].set_title('Total Loss')
    axes[0,0].legend()
    
    axes[0,1].plot(losses['recon_loss'], color='green')
    axes[0,1].set_title('Reconstruction Loss')
    
    axes[0,2].plot(losses['affine_loss'], color='orange')
    axes[0,2].set_title('Affine Loss')
    
    axes[1,0].plot(losses['content_reg'], color='purple')
    axes[1,0].set_title('Content Regularization')
    
    axes[1,1].plot(losses['transform_reg'], color='brown')
    axes[1,1].set_title('Transform Regularization')
    
    # Combined view
    axes[1,2].plot(losses['train_loss'], label='Total Train', alpha=0.7)
    axes[1,2].plot(losses['recon_loss'], label='Reconstruction', alpha=0.7)
    axes[1,2].plot(losses['affine_loss'], label='Affine', alpha=0.7)
    axes[1,2].set_title('Combined View')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.show()

def visualize_structured_latent_space(model, test_loader, device, num_samples=1000):
    """Visualize structured latent space with content and transform separation"""
    model.eval()
    
    content_latents = []
    transform_latents = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if len(content_latents) * test_loader.batch_size >= num_samples:
                break
                
            data = data.to(device)
            _, content_latent, transform_latent, _ = model(data)
            
            content_latents.append(content_latent.cpu().numpy())
            transform_latents.append(transform_latent.cpu().numpy())
            labels.append(label.numpy())
    
    content_data = np.concatenate(content_latents, axis=0)[:num_samples]
    transform_data = np.concatenate(transform_latents, axis=0)[:num_samples]
    label_data = np.concatenate(labels, axis=0)[:num_samples]
    
    # Plot structured latent space
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Content latent space (2D)
    scatter = axes[0].scatter(content_data[:, 0], content_data[:, 1], c=label_data, cmap='tab10', alpha=0.7)
    axes[0].set_title('Content Latent Space (2D)\nDigit Identity Clustering')
    axes[0].set_xlabel('Content Dimension 1')
    axes[0].set_ylabel('Content Dimension 2')
    plt.colorbar(scatter, ax=axes[0])
    
    # Transform latent space (first 2 of 6 dims)
    axes[1].scatter(transform_data[:, 0], transform_data[:, 1], c=label_data, cmap='tab10', alpha=0.7)
    axes[1].set_title('Transform Latent Space (Dims 1-2 of 6)\nSpatial Transformations')
    axes[1].set_xlabel('Transform Dimension 1')
    axes[1].set_ylabel('Transform Dimension 2')
    
    # Combined view
    total_latent = np.concatenate([content_data, transform_data], axis=1)
    axes[2].scatter(total_latent[:, 0], total_latent[:, 2], c=label_data, cmap='tab10', alpha=0.7)
    axes[2].set_title('Combined View\n(Content-1 vs Transform-1)')
    axes[2].set_xlabel('Content Dimension 1')
    axes[2].set_ylabel('Transform Dimension 1')
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Structured Latent Space Analysis:")
    print(f"  Content latent shape: {content_data.shape}")
    print(f"  Transform latent shape: {transform_data.shape}")
    print(f"  Content variance: {np.var(content_data, axis=0)}")
    print(f"  Transform variance: {np.var(transform_data, axis=0)}")
    
    return content_data, transform_data, label_data

def visualize_latent_space(model, test_loader, device, latent_dim):
    """Visualize standard latent space"""
    model.eval()
    
    latents = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if batch_idx >= 20:
                break
            data = data.to(device)
            _, _, latent, _ = model(data)
            latents.append(latent.cpu().numpy())
            labels.append(label.numpy())
    
    latent_data = np.concatenate(latents, axis=0)
    label_data = np.concatenate(labels, axis=0)
    
    # Plot first 2 dimensions
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_data[:, 0], latent_data[:, 1], c=label_data, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title(f'Latent Space Visualization ({latent_dim}D)')
    plt.show()
    
    print(f"📊 Latent space shape: {latent_data.shape}")
