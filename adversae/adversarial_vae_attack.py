"""
Adversarial Attacks on Variational Autoencoders
This script demonstrates how to engineer adversarial attacks against a VAE
using MNIST dataset with LeNet-style encoder/decoder and 2D latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class LeNetEncoder(nn.Module):
    """LeNet-style encoder for VAE"""
    def __init__(self, latent_dim=2):
        super(LeNetEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Convolutional layers (LeNet-style)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 14x14 -> 10x10
        self.pool2 = nn.MaxPool2d(2, 2)  # 10x10 -> 5x5
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(84, latent_dim)
        self.fc_logvar = nn.Linear(84, latent_dim)
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class LeNetDecoder(nn.Module):
    """LeNet-style decoder for VAE"""
    def __init__(self, latent_dim=2):
        super(LeNetDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim, 84)
        self.fc2 = nn.Linear(84, 120)
        self.fc3 = nn.Linear(120, 16 * 5 * 5)
        
        # Transposed convolutional layers (reverse of encoder)
        self.deconv1 = nn.ConvTranspose2d(16, 6, kernel_size=5, stride=2, padding=2, output_padding=1)  # 5x5 -> 10x10
        self.deconv2 = nn.ConvTranspose2d(6, 1, kernel_size=5, stride=2, padding=2, output_padding=1)   # 10x10 -> 20x20
        self.final_conv = nn.Conv2d(1, 1, kernel_size=9, padding=4)  # 20x20 -> 28x28
        
    def forward(self, z):
        # Fully connected layers
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Reshape to feature maps
        x = x.view(-1, 16, 5, 5)
        
        # Transposed convolutional layers
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.final_conv(x))
        
        return x


class VAE(nn.Module):
    """Variational Autoencoder with LeNet-style architecture"""
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = LeNetEncoder(latent_dim)
        self.decoder = LeNetDecoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    def encode(self, x):
        """Encode input to latent space"""
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function with KL divergence"""
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss


class AdversarialAttacks:
    """Class containing various adversarial attack methods for VAEs"""
    
    @staticmethod
    def fgsm_attack(model, data, target, epsilon):
        """
        Fast Gradient Sign Method (FGSM) attack
        
        Args:
            model: VAE model
            data: input data
            target: target data (for reconstruction loss)
            epsilon: perturbation magnitude
        """
        # Set model to evaluation mode
        model.eval()
        
        # Enable gradient computation for input
        data.requires_grad = True
        
        # Forward pass
        recon_data, mu, logvar = model(data)
        
        # Calculate loss
        loss = vae_loss(recon_data, target, mu, logvar)
        
        # Zero gradients
        model.zero_grad()
        
        # Calculate gradients
        loss.backward()
        
        # Get gradient sign
        data_grad = data.grad.data
        sign_data_grad = data_grad.sign()
        
        # Create adversarial example
        perturbed_data = data + epsilon * sign_data_grad
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        return perturbed_data
    
    @staticmethod
    def pgd_attack(model, data, target, epsilon, alpha, num_iter):
        """
        Projected Gradient Descent (PGD) attack
        
        Args:
            model: VAE model
            data: input data
            target: target data
            epsilon: maximum perturbation
            alpha: step size
            num_iter: number of iterations
        """
        model.eval()
        
        # Initialize perturbation
        delta = torch.zeros_like(data).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        
        for i in range(num_iter):
            # Forward pass with perturbation
            perturbed_data = data + delta
            recon_data, mu, logvar = model(perturbed_data)
            
            # Calculate loss
            loss = vae_loss(recon_data, target, mu, logvar)
            
            # Calculate gradients
            loss.backward()
            
            # Update perturbation
            delta.data = delta.data + alpha * delta.grad.data.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(data + delta.data, 0, 1) - data
            
            # Zero gradients
            delta.grad.zero_()
        
        return data + delta
    
    @staticmethod
    def latent_space_attack(model, data, epsilon, target_latent=None):
        """
        Attack in latent space by perturbing encoded representations
        
        Args:
            model: VAE model
            data: input data
            epsilon: perturbation magnitude in latent space
            target_latent: target latent representation (optional)
        """
        model.eval()
        
        # Encode to latent space
        mu, logvar = model.encoder(data)
        z = model.reparameterize(mu, logvar)
        
        if target_latent is not None:
            # Move towards target latent representation
            direction = (target_latent - z).sign()
            perturbed_z = z + epsilon * direction
        else:
            # Random perturbation in latent space
            noise = torch.randn_like(z)
            perturbed_z = z + epsilon * noise
        
        # Decode back to image space
        adversarial_recon = model.decoder(perturbed_z)
        
        return adversarial_recon, z, perturbed_z


def train_vae(model, train_loader, epochs=20, lr=1e-3, beta=1.0):
    """Train the VAE model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    train_losses = []
    
    print("Training VAE...")
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar, beta)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
    
    return train_losses


def visualize_attacks(original, fgsm_adv, pgd_adv, latent_adv, save_path=None):
    """Visualize original and adversarial examples"""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    # Convert tensors to numpy for visualization
    original = original.cpu().numpy()
    fgsm_adv = fgsm_adv.cpu().numpy()
    pgd_adv = pgd_adv.cpu().numpy()
    latent_adv = latent_adv.cpu().numpy()
    
    titles = ['Original', 'FGSM Attack', 'PGD Attack', 'Latent Attack']
    data_list = [original, fgsm_adv, pgd_adv, latent_adv]
    
    for i, (data, title) in enumerate(zip(data_list, titles)):
        # Show first example
        axes[0, i].imshow(data[0, 0], cmap='gray')
        axes[0, i].set_title(f'{title}')
        axes[0, i].axis('off')
        
        # Show second example if available
        if len(data) > 1:
            axes[1, i].imshow(data[1, 0], cmap='gray')
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_latent_space(model, test_loader, device, save_path=None):
    """Plot the 2D latent space representation"""
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            mu, _ = model.encoder(data)
            latents.append(mu.cpu().numpy())
            labels.append(label.numpy())
    
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('2D Latent Space Representation')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_attack_effectiveness(model, original, adversarial, device):
    """Evaluate the effectiveness of adversarial attacks"""
    model.eval()
    
    with torch.no_grad():
        # Reconstruct original
        original = original.to(device)
        recon_orig, mu_orig, logvar_orig = model(original)
        
        # Reconstruct adversarial
        adversarial = adversarial.to(device)
        recon_adv, mu_adv, logvar_adv = model(adversarial)
        
        # Calculate reconstruction errors
        orig_error = F.mse_loss(recon_orig, original).item()
        adv_error = F.mse_loss(recon_adv, adversarial).item()
        
        # Calculate latent space distances
        latent_distance = F.mse_loss(mu_orig, mu_adv).item()
        
        # Calculate input perturbation
        input_perturbation = F.mse_loss(original, adversarial).item()
        
        print(f"Original Reconstruction Error: {orig_error:.6f}")
        print(f"Adversarial Reconstruction Error: {adv_error:.6f}")
        print(f"Latent Space Distance: {latent_distance:.6f}")
        print(f"Input Perturbation (MSE): {input_perturbation:.6f}")
        
        return orig_error, adv_error, latent_distance, input_perturbation


def main():
    """Main function to demonstrate adversarial attacks on VAE"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize VAE model
    model = VAE(latent_dim=2)
    
    # Train the model
    train_losses = train_vae(model, train_loader, epochs=20, beta=1.0)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # Visualize latent space
    plot_latent_space(model, test_loader, device)
    
    # Get test samples for attacks
    test_iter = iter(test_loader)
    test_data, test_labels = next(test_iter)
    test_data = test_data[:4]  # Use first 4 samples
    
    # Move to device
    test_data = test_data.to(device)
    
    # Initialize attack methods
    attacks = AdversarialAttacks()
    
    # FGSM Attack
    print("\nPerforming FGSM Attack...")
    fgsm_adversarial = attacks.fgsm_attack(model, test_data, test_data, epsilon=0.1)
    evaluate_attack_effectiveness(model, test_data, fgsm_adversarial, device)
    
    # PGD Attack
    print("\nPerforming PGD Attack...")
    pgd_adversarial = attacks.pgd_attack(model, test_data, test_data, 
                                       epsilon=0.1, alpha=0.01, num_iter=20)
    evaluate_attack_effectiveness(model, test_data, pgd_adversarial, device)
    
    # Latent Space Attack
    print("\nPerforming Latent Space Attack...")
    latent_adversarial, orig_latent, perturbed_latent = attacks.latent_space_attack(
        model, test_data, epsilon=2.0)
    
    # For latent attack, we compare reconstructions
    with torch.no_grad():
        original_recon, _, _ = model(test_data)
    
    print(f"Original vs Latent Attack Reconstruction MSE: {F.mse_loss(original_recon, latent_adversarial).item():.6f}")
    
    # Visualize all attacks
    print("\nVisualizing attacks...")
    visualize_attacks(test_data, fgsm_adversarial, pgd_adversarial, latent_adversarial)
    
    # Demonstrate robustness across different epsilon values
    print("\nTesting robustness across different epsilon values...")
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    fgsm_errors = []
    
    for eps in epsilons:
        fgsm_adv = attacks.fgsm_attack(model, test_data, test_data, epsilon=eps)
        _, adv_error, _, _ = evaluate_attack_effectiveness(model, test_data, fgsm_adv, device)
        fgsm_errors.append(adv_error)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, fgsm_errors, 'o-')
    plt.xlabel('Epsilon (Perturbation Magnitude)')
    plt.ylabel('Reconstruction Error')
    plt.title('VAE Robustness vs Adversarial Perturbation Magnitude')
    plt.grid(True)
    plt.show()
    
    print("\nAdversarial attack demonstration completed!")


if __name__ == "__main__":
    main()
