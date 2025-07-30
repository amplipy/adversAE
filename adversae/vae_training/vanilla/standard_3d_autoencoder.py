"""
Standard 3D Autoencoder - Self-Contained Module
Vanilla autoencoder with 3D latent space for MNIST digit reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class StandardAutoEncoder(nn.Module):
    """Variational Autoencoder with 3D latent space and KL regularization"""
    def __init__(self, latent_dim=3):
        super(StandardAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 28x28 -> mu, logvar
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),   # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: 3D latent -> 28x28
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),   # 14x14 -> 28x28
            nn.Sigmoid()
        )
        
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for differentiable sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent variable to reconstruction"""
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar, z


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function with KL regularization
    
    Args:
        recon_x: Reconstructed images
        x: Original images  
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (normalized by image dimensions)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    
    # KL divergence loss (normalized by batch size and latent dimensions)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def get_mnist_loaders(batch_size_train=128, batch_size_test=64, data_dir='./data'):
    """Load MNIST dataset"""
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, 
                                               download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, 
                                              transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)
    
    return train_loader, test_loader


def train_autoencoder(model, train_loader, epochs, lr, device=None, beta=1.0):
    """Train the VAE model"""
    if device is None:
        device = get_device()
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = {'total': [], 'reconstruction': [], 'kl': []}
    
    model.train()
    for epoch in range(epochs):
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, mu, logvar, z = model(data)
            
            # VAE loss with KL regularization
            total_loss, recon_loss, kl_loss = vae_loss(reconstruction, data, mu, logvar, beta)
            
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}'
            })
        
        # Store epoch averages
        num_batches = len(train_loader)
        losses['total'].append(epoch_total_loss / num_batches)
        losses['reconstruction'].append(epoch_recon_loss / num_batches)
        losses['kl'].append(epoch_kl_loss / num_batches)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Total Loss: {losses["total"][-1]:.4f}, '
              f'Recon Loss: {losses["reconstruction"][-1]:.4f}, '
              f'KL Loss: {losses["kl"][-1]:.4f}')
    
    return losses


def evaluate_model(model, test_loader, device=None, beta=1.0):
    """Evaluate VAE model on test set"""
    if device is None:
        device = get_device()
    
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstruction, mu, logvar, z = model(data)
            loss, recon_loss, kl_loss = vae_loss(reconstruction, data, mu, logvar, beta)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(test_loader)
    avg_recon_loss = total_recon_loss / len(test_loader)
    avg_kl_loss = total_kl_loss / len(test_loader)
    
    print(f"Test - Total: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}")
    return avg_loss, avg_recon_loss, avg_kl_loss


def visualize_reconstructions(model, test_loader, device=None, n_samples=8):
    """Visualize original vs reconstructed images"""
    if device is None:
        device = get_device()
    
    model.eval()
    test_iter = iter(test_loader)
    data, labels = next(test_iter)
    data = data[:n_samples].to(device)
    labels = labels[:n_samples]
    
    with torch.no_grad():
        reconstruction, mu, logvar, z = model(data)
    
    fig, axes = plt.subplots(4, n_samples, figsize=(n_samples * 1.5, 6))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original\n{labels[i].item()}')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstruction[i].cpu().squeeze(), cmap='gray')
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
        
        # Latent mean (mu)
        mu_vals = mu[i].cpu().numpy()
        axes[2, i].bar(range(3), mu_vals, color=['red', 'green', 'blue'])
        axes[2, i].set_title('μ (Mean)')
        axes[2, i].set_xticks([0, 1, 2])
        axes[2, i].set_xticklabels(['X', 'Y', 'Z'])
        axes[2, i].set_ylim(mu_vals.min()-0.5, mu_vals.max()+0.5)
        
        # Latent std (from logvar)
        std_vals = torch.exp(0.5 * logvar[i]).cpu().numpy()
        axes[3, i].bar(range(3), std_vals, color=['red', 'green', 'blue'], alpha=0.7)
        axes[3, i].set_title('σ (Std)')
        axes[3, i].set_xticks([0, 1, 2])
        axes[3, i].set_xticklabels(['X', 'Y', 'Z'])
        axes[3, i].set_ylim(0, std_vals.max()+0.1)
    
    plt.suptitle('3D VAE Results (μ ± σ)')
    plt.tight_layout()
    plt.show()


def visualize_3d_latent_space(model, test_loader, device=None, n_samples=1000):
    """Visualize 3D latent space"""
    if device is None:
        device = get_device()
    
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            if len(latents) * test_loader.batch_size >= n_samples:
                break
            data = data.to(device)
            _, mu, logvar, z = model(data)
            latents.append(mu.cpu().numpy())  # Use mean for visualization
            labels.append(label.numpy())
    
    latent_data = np.concatenate(latents, axis=0)[:n_samples]
    label_data = np.concatenate(labels, axis=0)[:n_samples]
    
    # 3D scatter plot
    fig = plt.figure(figsize=(15, 5))
    
    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(latent_data[:, 0], latent_data[:, 1], latent_data[:, 2], 
                         c=label_data, cmap='tab10', alpha=0.6, s=20)
    ax1.set_xlabel('Latent X (μ₁)')
    ax1.set_ylabel('Latent Y (μ₂)')
    ax1.set_zlabel('Latent Z (μ₃)')
    ax1.set_title('3D VAE Latent Space (μ)')
    
    # 2D projections
    ax2 = fig.add_subplot(132)
    ax2.scatter(latent_data[:, 0], latent_data[:, 1], c=label_data, cmap='tab10', alpha=0.6, s=20)
    ax2.set_xlabel('Latent X (μ₁)')
    ax2.set_ylabel('Latent Y (μ₂)')
    ax2.set_title('XY Projection')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(133)
    ax3.scatter(latent_data[:, 0], latent_data[:, 2], c=label_data, cmap='tab10', alpha=0.6, s=20)
    ax3.set_xlabel('Latent X (μ₁)')
    ax3.set_ylabel('Latent Z (μ₃)')
    ax3.set_title('XZ Projection')
    ax3.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=[ax2, ax3])
    plt.tight_layout()
    plt.show()
    
    print(f"3D VAE Latent Space Stats (μ):")
    print(f"  Shape: {latent_data.shape}")
    print(f"  Range: X[{latent_data[:, 0].min():.2f}, {latent_data[:, 0].max():.2f}]")
    print(f"         Y[{latent_data[:, 1].min():.2f}, {latent_data[:, 1].max():.2f}]")
    print(f"         Z[{latent_data[:, 2].min():.2f}, {latent_data[:, 2].max():.2f}]")
    
    return latent_data, label_data


def latent_grid_interpolation(model, device=None, grid_size=10, latent_range=(-2, 2)):
    """Generate images by interpolating through 3D latent space"""
    if device is None:
        device = get_device()
    
    model.eval()
    
    # Create grid for first two dimensions, fix third at 0
    x = np.linspace(latent_range[0], latent_range[1], grid_size)
    y = np.linspace(latent_range[0], latent_range[1], grid_size)
    
    latent_vectors = torch.zeros(grid_size * grid_size, 3, device=device)
    
    for i, y_val in enumerate(y):
        for j, x_val in enumerate(x):
            idx = i * grid_size + j
            latent_vectors[idx, 0] = x_val
            latent_vectors[idx, 1] = y_val
            latent_vectors[idx, 2] = 0.0  # Fix Z dimension
    
    with torch.no_grad():
        images = model.decode(latent_vectors)  # Use decode method
    
    # Visualize grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 0.8, grid_size * 0.8))
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            img = images[idx].cpu().squeeze()
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
    
    plt.suptitle(f'3D VAE Latent Space Grid (Z=0)\nX: {latent_range[0]} to {latent_range[1]}, Y: {latent_range[0]} to {latent_range[1]}')
    plt.tight_layout()
    plt.show()
    
    return images.cpu().numpy()


def save_model(model, filepath, config=None, losses=None):
    """Save model and metadata"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'losses': losses,
        'model_class': 'StandardAutoEncoder'
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device=None):
    """Load model from file"""
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model
    config = checkpoint.get('config', {})
    latent_dim = config.get('latent_dim', 3)
    model = StandardAutoEncoder(latent_dim=latent_dim)
    
    # Load state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, checkpoint.get('config'), checkpoint.get('losses')


def plot_training_losses(losses):
    """Plot VAE training loss curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(losses['total']) + 1)
    
    # Plot all losses on left subplot
    ax1.plot(epochs, losses['total'], 'b-', linewidth=2, label='Total Loss')
    ax1.plot(epochs, losses['reconstruction'], 'r--', linewidth=2, label='Reconstruction')
    ax1.plot(epochs, losses['kl'], 'g:', linewidth=2, label='KL Divergence')
    ax1.set_title('VAE Training Losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot total loss only on right subplot for clarity
    ax2.plot(epochs, losses['total'], 'b-', linewidth=2)
    ax2.set_title('Total Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Total Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Final losses - Total: {losses['total'][-1]:.6f}, "
          f"Reconstruction: {losses['reconstruction'][-1]:.6f}, "
          f"KL: {losses['kl'][-1]:.6f}")
    print(f"Best total loss: {min(losses['total']):.6f} "
          f"(epoch {losses['total'].index(min(losses['total'])) + 1})")
