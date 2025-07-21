"""
Structured 2D+6D Affine-Invariant Autoencoder
This module contains classes and functions specific to the structured 2D content + 6D transform autoencoder.
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
        
        # Content encoder (digit identity) - VAE style with mu and logvar
        self.content_mu = nn.Linear(512, content_dim)
        self.content_logvar = nn.Linear(512, content_dim)
        
        # Transform encoder (spatial transformations) - deterministic
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
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Shared encoding
        shared_features = self.shared_encoder(x)
        
        # Content latent with VAE reparameterization
        content_mu = self.content_mu(shared_features)
        content_logvar = self.content_logvar(shared_features)
        content_latent = self.reparameterize(content_mu, content_logvar)
        
        # Transform latent (deterministic)
        transform_latent = self.transform_encoder(shared_features)
        
        # Combine and decode
        combined_latent = torch.cat([content_latent, transform_latent], dim=1)
        reconstruction = self.decoder(combined_latent)
        
        return reconstruction, content_latent, transform_latent, content_mu, content_logvar


class StructuredAffineInvariantAutoEncoder(nn.Module):
    """Combined structured autoencoder with affine transformation network"""
    def __init__(self, content_dim=2, transform_dim=6):
        super(StructuredAffineInvariantAutoEncoder, self).__init__()
        
        self.structured_autoencoder = StructuredAutoEncoder(content_dim, transform_dim)
        self.affine_net = shared.AffineTransformationNetwork()
    
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
        # Get structured reconstruction with VAE outputs
        reconstruction, content_latent, transform_latent, content_mu, content_logvar = self.structured_autoencoder(x)
        
        # Predict affine transformation parameters
        affine_params = self.affine_net(x)
        
        # Apply affine transformation to reconstruction
        transformed_reconstruction = self.apply_affine_transformation(reconstruction, affine_params)
        
        return reconstruction, content_latent, transform_latent, affine_params, transformed_reconstruction, content_mu, content_logvar


def train_structured_autoencoder_simplified(model, train_loader, test_loader, device, config, scaler=None):
    """Training for structured autoencoder with simplified affine+KL loss"""
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                          weight_decay=config.get('weight_decay', 1e-5))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7) if config.get('lr_scheduler', False) else None
    
    # Use simplified loss components
    losses = {'train_loss': [], 'test_loss': [], 'recon_loss': [], 'kl_loss': []}
    
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        recon_total = 0
        kl_total = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data = data.to(device)
            
            with torch.cuda.amp.autocast() if scaler else torch.enable_grad():
                # Forward pass - get all outputs including mu and logvar
                (reconstruction, content_latent, transform_latent, affine_params, 
                 transformed_reconstruction, content_mu, content_logvar) = model(data)
                
                # Use simplified loss: affine-transformed reconstruction + KL divergence
                total_loss, recon_loss, kl_loss = shared.simplified_affine_kl_loss(
                    data, transformed_reconstruction, content_mu, content_logvar,
                    alpha=config.get('alpha', 1.0), 
                    beta=config.get('beta', 0.001)
                )
            
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
            kl_total += kl_loss.item()
        
        # Testing
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                (reconstruction, content_latent, transform_latent, affine_params, 
                 transformed_reconstruction, content_mu, content_logvar) = model(data)
                
                total_loss, recon_loss, kl_loss = shared.simplified_affine_kl_loss(
                    data, transformed_reconstruction, content_mu, content_logvar,
                    alpha=config.get('alpha', 1.0), 
                    beta=config.get('beta', 0.001)
                )
                test_loss += total_loss.item()
        
        # Record losses
        avg_train = train_loss / len(train_loader)
        avg_test = test_loss / len(test_loader)
        losses['train_loss'].append(avg_train)
        losses['test_loss'].append(avg_test)
        losses['recon_loss'].append(recon_total / len(train_loader))
        losses['kl_loss'].append(kl_total / len(train_loader))
        
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
        
        print(f"Epoch {epoch+1}: Train={avg_train:.6f}, Test={avg_test:.6f}, Recon={recon_total/len(train_loader):.6f}, KL={kl_total/len(train_loader):.6f}")
    
    return losses


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
            (_, content_latent, transform_latent, _, _, _, _) = model(data)
            
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


def comprehensive_visualization_structured(model, test_loader, device, config):
    """
    Comprehensive visualization for structured affine+KL loss model.
    Shows the key insight: affine transformation bridges reconstruction and original.
    """
    model.eval()
    
    # Get test samples
    test_iter = iter(test_loader)
    test_images, test_labels = next(test_iter)
    samples = test_images[:8].to(device)
    labels = test_labels[:8]
    
    with torch.no_grad():
        # Get all model outputs
        (reconstruction, content_latent, transform_latent, affine_params, 
         transformed_reconstruction, content_mu, content_logvar) = model(samples)
        
        # Compute loss components for analysis
        total_loss, recon_loss, kl_loss = shared.simplified_affine_kl_loss(
            samples, transformed_reconstruction, content_mu, content_logvar,
            alpha=config.get('alpha', 1.0), beta=config.get('beta', 0.001)
        )
    
    # Create visualization showing the key insight
    fig, axes = plt.subplots(5, 8, figsize=(16, 10))
    
    for i in range(8):
        # Row 1: Original images
        axes[0, i].imshow(samples[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original\nDigit: {labels[i].item()}', fontsize=10)
        axes[0, i].axis('off')
        
        # Row 2: Direct reconstruction (before affine correction)
        axes[1, i].imshow(reconstruction[i].cpu().squeeze(), cmap='gray')
        axes[1, i].set_title('Raw Decoder\nOutput', fontsize=10)
        axes[1, i].axis('off')
        
        # Row 3: After affine transformation (what the loss function optimizes)
        axes[2, i].imshow(transformed_reconstruction[i].cpu().squeeze(), cmap='gray')
        axes[2, i].set_title('After Affine\n(Loss Target)', fontsize=10)
        axes[2, i].axis('off')
        
        # Row 4: Content latent space (2D) with actual values
        content_vals = content_latent[i].cpu().numpy()
        axes[3, i].scatter([0], [0], s=100, c='red', marker='x')
        axes[3, i].text(0.1, 0.7, f'c1={content_vals[0]:.2f}', transform=axes[3, i].transAxes, fontsize=8)
        axes[3, i].text(0.1, 0.5, f'c2={content_vals[1]:.2f}', transform=axes[3, i].transAxes, fontsize=8)
        axes[3, i].set_xlim(-0.5, 0.5)
        axes[3, i].set_ylim(-0.5, 0.5)
        axes[3, i].set_title('2D Content\nLatent', fontsize=10)
        
        # Row 5: Affine parameters
        params = affine_params[i].cpu().numpy()
        axes[4, i].text(0.05, 0.8, f'a={params[0]:.2f}', transform=axes[4, i].transAxes, fontsize=8)
        axes[4, i].text(0.05, 0.6, f'b={params[1]:.2f}', transform=axes[4, i].transAxes, fontsize=8)
        axes[4, i].text(0.05, 0.4, f'c={params[2]:.2f}', transform=axes[4, i].transAxes, fontsize=8)
        axes[4, i].text(0.55, 0.8, f'd={params[3]:.2f}', transform=axes[4, i].transAxes, fontsize=8)
        axes[4, i].text(0.55, 0.6, f'tx={params[4]:.2f}', transform=axes[4, i].transAxes, fontsize=8)
        axes[4, i].text(0.55, 0.4, f'ty={params[5]:.2f}', transform=axes[4, i].transAxes, fontsize=8)
        axes[4, i].set_title('Affine Params', fontsize=10)
        axes[4, i].axis('off')
    
    plt.suptitle('🎯 Structured 2D+6D Affine+KL Loss Model\nKey: Affine transforms decoder output into clean digits!', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Print loss analysis
    print(f"\n📊 Loss Analysis (Simplified Affine+KL Loss Function):")
    print(f"  Total Loss: {total_loss.item():.6f}")
    print(f"  Reconstruction Loss (MSE after affine): {recon_loss.item():.6f}")
    print(f"  KL Divergence Loss (2D content latent): {kl_loss.item():.6f}")
    print(f"  Loss weights: α={config.get('alpha', 1.0)} (recon), β={config.get('beta', 0.001)} (KL)")
    
    # Analyze content latent space
    print(f"\n🎨 2D Content Latent Analysis:")
    print(f"  Content latent shape: {content_latent.shape}")
    print(f"  Content mu mean: {content_mu.mean(dim=0).cpu().numpy()}")
    print(f"  Content mu std: {content_mu.std(dim=0).cpu().numpy()}")
    print(f"  Content logvar mean: {content_logvar.mean(dim=0).cpu().numpy()}")
    
    print(f"\n🔧 6D Transform Latent Analysis:")
    print(f"  Transform latent shape: {transform_latent.shape}")
    print(f"  Transform mean: {transform_latent.mean(dim=0).cpu().numpy()}")
    print(f"  Transform std: {transform_latent.std(dim=0).cpu().numpy()}")


def scan_2d_content_latent_grid(model, device, grid_size=10, latent_range=(-3, 3)):
    """Scan through 2D content latent space and show decoder output"""
    model.eval()
    
    # Create a grid of content latent coordinates  
    x = np.linspace(latent_range[0], latent_range[1], grid_size)
    y = np.linspace(latent_range[0], latent_range[1], grid_size)
    
    # Initialize latent vectors: 2D content + 6D transform (set to 0)
    content_vectors = torch.zeros(grid_size * grid_size, 2, device=device)
    transform_vectors = torch.zeros(grid_size * grid_size, 6, device=device)
    
    # Fill in the 2D content dimensions with grid coordinates
    for i, y_val in enumerate(y):
        for j, x_val in enumerate(x):
            idx = i * grid_size + j
            content_vectors[idx, 0] = x_val  # Content dimension 1
            content_vectors[idx, 1] = y_val  # Content dimension 2
    
    # Generate images from latent vectors
    with torch.no_grad():
        # Combine content and transform latents
        combined_latent = torch.cat([content_vectors, transform_vectors], dim=1)
        raw_images = model.structured_autoencoder.decoder(combined_latent)
    
    # Create visualization
    images = raw_images.cpu().numpy()
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 1.2, grid_size * 1.2))
    
    if grid_size == 1:
        axes = [axes]
    
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
            
            # Add coordinates for key points
            if (i == 0 and j == 0) or (i == grid_size//2 and j == grid_size//2) or (i == grid_size-1 and j == grid_size-1):
                ax.set_title(f'({x[j]:.1f},{y[i]:.1f})', fontsize=8)
    
    plt.suptitle('2D Content Latent Space Grid Scan\n(Transform Latent = 0)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 2D Content Latent Grid Analysis:")
    print(f"  Grid size: {grid_size} × {grid_size} = {grid_size*grid_size} generated images")
    print(f"  Content latent dimensions scanned: 2D (full space)")
    print(f"  Transform latent: 6D (all set to 0)")
    print(f"  Scan range: {latent_range[0]} to {latent_range[1]} for both content dimensions")
    
    return images, content_vectors.cpu().numpy(), transform_vectors.cpu().numpy()


def plot_simplified_training_progress_structured(losses):
    """Plot training progress for simplified affine+KL loss"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Total loss
    axes[0].plot(losses['train_loss'], label='Train', color='blue', linewidth=2)
    axes[0].plot(losses['test_loss'], label='Test', color='red', linewidth=2)
    axes[0].set_title('Total Loss\n(Affine Reconstruction + KL)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss (MSE after affine)
    axes[1].plot(losses['recon_loss'], color='green', linewidth=2)
    axes[1].set_title('Reconstruction Loss\n(MSE after Affine Transform)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    
    # KL divergence loss
    axes[2].plot(losses['kl_loss'], color='purple', linewidth=2)
    axes[2].set_title('KL Divergence Loss\n(2D Content Latent)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('🎯 Structured 2D+6D Simplified Loss Function Training Progress', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print final loss values
    print(f"\n📊 Final Loss Values:")
    print(f"  Total Train Loss: {losses['train_loss'][-1]:.6f}")
    print(f"  Total Test Loss: {losses['test_loss'][-1]:.6f}")
    print(f"  Reconstruction Loss: {losses['recon_loss'][-1]:.6f}")
    print(f"  KL Divergence Loss: {losses['kl_loss'][-1]:.6f}")
