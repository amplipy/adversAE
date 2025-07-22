"""
Affine-Invariant Autoencoder for MNIST - Shared Components
This module contains shared classes and functions used by both standard 64D and structured 2D+6D autoencoders.

Includes:
- AffineTransformationNetwork (shared by both model types)
- Loss functions (original and simplified)
- Data loading utilities
- Device management
- Model loading/saving utilities
- Common visualization functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
from matplotlib.patches import Patch
from tqdm import tqdm
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


# =============================================================================
# SHARED LOSS FUNCTIONS
# =============================================================================

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


def simplified_affine_kl_loss(original, transformed_reconstruction, content_latent_mu, content_latent_logvar, 
                             alpha=1.0, beta=0.001):
    """
    Simplified loss function: affine-transformed reconstruction + KL divergence (PROPERLY NORMALIZED).
    
    Args:
        original: Original input images [batch_size, 1, 28, 28]
        transformed_reconstruction: Affine-corrected reconstruction [batch_size, 1, 28, 28]
        content_latent_mu: Mean of latent distribution [batch_size, latent_dim]
        content_latent_logvar: Log variance of latent distribution [batch_size, latent_dim]
        alpha: Weight for reconstruction loss (typically 1.0)
        beta: Weight for KL divergence (typically 0.001-0.01, depends on latent_dim)
    
    Returns:
        Total loss, reconstruction loss, KL divergence loss
    """
    batch_size = original.size(0)
    latent_dim = content_latent_mu.size(1)
    
    # Reconstruction loss: MSE between affine-corrected reconstruction and original
    # Normalize by number of pixels for stability
    recon_loss = F.mse_loss(transformed_reconstruction, original, reduction='sum')
    recon_loss = recon_loss / (batch_size * original.numel() / batch_size)  # Per pixel
    
    # KL divergence loss - PROPERLY NORMALIZED
    # KL(q(z|x) || p(z)) where p(z) = N(0,I) and q(z|x) = N(mu, sigma^2)
    kl_loss = -0.5 * torch.sum(1 + content_latent_logvar - content_latent_mu.pow(2) - content_latent_logvar.exp())
    kl_loss = kl_loss / batch_size  # Per sample, per dimension
    
    # Combined loss
    total_loss = alpha * recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss
    
    return total_loss, recon_loss, kl_loss


# =============================================================================
# SHARED DATA LOADING FUNCTIONS
# =============================================================================

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


# =============================================================================
# SHARED DEVICE MANAGEMENT
# =============================================================================

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


# =============================================================================
# SHARED MODEL LOADING/SAVING FUNCTIONS
# =============================================================================

def save_model(model, filepath):
    """Save model state dict"""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device):
    """Load model state dict"""
    model.load_state_dict(torch.load(filepath, map_location=device, weights_only=False))
    model.to(device)
    print(f"Model loaded from {filepath}")
    return model


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


def list_saved_models(pattern="*model*.pth"):
    """List all saved model files in current directory"""
    
    model_files = glob.glob(pattern)
    model_files.sort(reverse=True)  # Most recent first
    
    if not model_files:
        print("📁 No saved models found")
        return []
    
    print(f"📁 Found {len(model_files)} saved models:")
    for i, file in enumerate(model_files):
        # Try to get timestamp from filename
        try:
            timestamp_part = file.split('_')[-1].replace('.pth', '')
            if len(timestamp_part) == 15:  # YYYYMMDD_HHMMSS format
                formatted_time = f"{timestamp_part[:8]}_{timestamp_part[9:]}"
                print(f"  {i+1}. {file} ({formatted_time})")
            else:
                print(f"  {i+1}. {file}")
        except:
            print(f"  {i+1}. {file}")
    
    return model_files


# =============================================================================
# SHARED TRAINING PROGRESS VISUALIZATION
# =============================================================================

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


def plot_simplified_training_progress(losses):
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
    
    plt.suptitle('🎯 Simplified Loss Function Training Progress', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print final loss values
    print(f"\n📊 Final Loss Values:")
    print(f"  Total Train Loss: {losses['train_loss'][-1]:.6f}")
    print(f"  Total Test Loss: {losses['test_loss'][-1]:.6f}")
    print(f"  Reconstruction Loss: {losses['recon_loss'][-1]:.6f}")
    print(f"  KL Divergence Loss: {losses['kl_loss'][-1]:.6f}")


# =============================================================================
# SHARED GENERAL VISUALIZATION FUNCTIONS  
# =============================================================================

def show_random_reconstructions(model, test_loader, device, num_samples=8):
    """Show random digit reconstructions with original and reconstruction"""
    model.eval()
    
    # Get random test samples
    test_iter = iter(test_loader)
    test_images, test_labels = next(test_iter)
    
    # Select random samples
    indices = torch.randperm(len(test_images))[:num_samples]
    samples = test_images[indices].to(device)
    labels = test_labels[indices]
    
    with torch.no_grad():
        # This is a generic function - specific models will override the forward call
        model_output = model(samples)
        
        # Handle different model output formats
        if len(model_output) >= 4:
            # Models with affine outputs
            reconstructed = model_output[0]
            if len(model_output) > 4:
                # Structured model might have affine-corrected version
                try:
                    corrected = model_output[4]  # transformed_reconstruction
                except:
                    corrected = reconstructed
            else:
                # Standard model
                corrected = reconstructed
        else:
            reconstructed = model_output[0]
            corrected = reconstructed
    
    # Visualize results
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(samples[i].cpu().squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original\nDigit: {labels[i].item()}')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
        
        # Corrected (if available)
        axes[2, i].imshow(corrected[i].cpu().squeeze(), cmap='gray')
        axes[2, i].set_title('Final Output')
        axes[2, i].axis('off')
    
    plt.suptitle('🎲 Random Digit Reconstructions', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print(f"🎯 Reconstructed {num_samples} random samples")


# =============================================================================
# VISUALIZATION-FOCUSED SAVE/LOAD FUNCTIONS
# =============================================================================

def save_model_for_viz(model, model_type, config=None, losses=None, extra_data=None, name=None):
    """
    Save model with all data needed for visualization
    
    Args:
        model: The trained model
        model_type: 'standard' or 'structured'
        config: Training configuration dict
        losses: Training losses dict
        extra_data: Additional data (latent embeddings, etc.)
        name: Optional custom name, otherwise uses timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if name:
        filename = f"{model_type}_{name}_{timestamp}.pth"
    else:
        filename = f"{model_type}_model_{timestamp}.pth"
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'timestamp': timestamp,
    }
    
    if config:
        save_dict['config'] = config
    if losses:
        save_dict['losses'] = losses
    if extra_data:
        save_dict['extra_data'] = extra_data
    
    torch.save(save_dict, filename)
    print(f"💾 Saved {model_type} model for visualization: {filename}")
    return filename


def load_model_for_viz(filename, model_class, device=None):
    """
    Load model and all visualization data
    
    Args:
        filename: Path to saved model file
        model_class: Model class to instantiate (e.g., s2d6d.StructuredAffineInvariantAutoEncoder)
        device: Target device, auto-detected if None
        
    Returns:
        model: Loaded model
        data_dict: Dict containing config, losses, extra_data, etc.
    """
    if device is None:
        device = get_device()
    
    # Load with weights_only=False to handle numpy data and config objects
    checkpoint = torch.load(filename, map_location=device, weights_only=False)
    
    # Extract model parameters from state dict to recreate model
    state_dict = checkpoint['model_state_dict']
    model_type = checkpoint.get('model_type', 'unknown')
    
    # Try to infer model parameters for structured models
    if 'structured' in model_type.lower():
        # Look for content/transform dimensions in the saved config
        config = checkpoint.get('config', {})
        content_dim = config.get('content_latent_dim', 2)
        transform_dim = config.get('transform_latent_dim', 6)
        model = model_class(content_dim=content_dim, transform_dim=transform_dim)
    else:
        # Standard model - try to infer latent_dim
        config = checkpoint.get('config', {})
        latent_dim = config.get('latent_dim', 64)
        model = model_class(latent_dim=latent_dim)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    data_dict = {
        'config': checkpoint.get('config', {}),
        'losses': checkpoint.get('losses', {}),
        'extra_data': checkpoint.get('extra_data', {}),
        'timestamp': checkpoint.get('timestamp', 'unknown'),
        'model_type': model_type,
        'device': device
    }
    
    print(f"📁 Loaded {model_type} model from {filename}")
    print(f"   Timestamp: {data_dict['timestamp']}")
    print(f"   Device: {device}")
    
    return model, data_dict


def quick_save_viz(model, model_type, name="latest"):
    """Quick save for visualization - just model state"""
    return save_model_for_viz(model, model_type, name=name)


def quick_load_viz(model_class, model_type="structured", name="latest", device=None):
    """
    Quick load the most recent model of a given type
    
    Args:
        model_class: Model class to instantiate
        model_type: 'standard' or 'structured'
        name: Model name pattern to search for
        device: Target device
    """
    # Find most recent model file matching pattern
    pattern = f"{model_type}*{name}*.pth"
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"❌ No models found matching pattern: {pattern}")
        return None, None
    
    # Sort by modification time, most recent first
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = model_files[0]
    
    return load_model_for_viz(latest_file, model_class, device)


if __name__ == "__main__":
    print("This is the shared components module for affine-invariant autoencoders.")
    print("Import specific model classes from:")
    print("  - standard_64d_autoencoder.py for 64D models")
    print("  - structured_2d6d_autoencoder.py for 2D+6D models")
