# affine_adversarial_attacks.py
"""
Adversarial attack analysis module for Structured Affine Autoencoder.

This module contains all functions for attacking and analyzing the structured 2D+6D affine autoencoder:
- FGSM, PGD, and latent space attacks adapted for unified 8D VAE
- Visualization functions showing content vs transform latent effects
- Attack effectiveness analysis on both reconstruction and affine transformation
- Statistical analysis specialized for affine-invariant models
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import random
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Import shared utilities and model
from affine_autoencoder_shared import *
import structured_2d6d_autoencoder as s2d6d


class AffineAdversarialAttacks:
    """Adversarial attacks specialized for Structured Affine Autoencoder"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def fgsm_attack(self, images, labels, epsilon):
        """Fast Gradient Sign Method attack on affine autoencoder"""
        images = images.clone().detach().requires_grad_(True)
        
        # Forward pass to get all outputs from unified 8D VAE
        (input_x, content_latent, transform_latent, unused, 
         clean_reconstruction, latent_mu, latent_logvar, final_reconstruction) = self.model(images)
        
        # Calculate loss using the simplified affine KL loss
        loss, recon_loss, kl_loss = simplified_affine_kl_loss(
            images, final_reconstruction, latent_mu, latent_logvar,
            alpha=1.0, beta=0.008
        )
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Collect gradients and create adversarial examples
        data_grad = images.grad.data
        perturbed_data = images + epsilon * data_grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        return perturbed_data.detach()
    
    def pgd_attack(self, images, labels, epsilon, alpha=0.01, num_iter=10):
        """Projected Gradient Descent attack on affine autoencoder"""
        images_orig = images.clone()
        images = images.clone().detach().requires_grad_(True)
        
        for i in range(num_iter):
            images.requires_grad_()
            
            # Forward pass
            (input_x, content_latent, transform_latent, unused, 
             clean_reconstruction, latent_mu, latent_logvar, final_reconstruction) = self.model(images)
            
            # Calculate loss
            loss, recon_loss, kl_loss = simplified_affine_kl_loss(
                images, final_reconstruction, latent_mu, latent_logvar,
                alpha=1.0, beta=0.008
            )
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            data_grad = images.grad.data
            images = images + alpha * data_grad.sign()
            
            # Project back to epsilon ball
            perturbation = torch.clamp(images - images_orig, -epsilon, epsilon)
            images = torch.clamp(images_orig + perturbation, 0, 1).detach()
        
        return images
    
    def latent_space_attack(self, images, labels, epsilon):
        """Attack in the latent space - manipulate content vs transform separately"""
        images = images.clone().detach().requires_grad_(True)
        
        # Get latent representations
        with torch.no_grad():
            (input_x, content_latent, transform_latent, unused, 
             clean_reconstruction, latent_mu, latent_logvar, final_reconstruction) = self.model(images)
        
        # Create adversarial latent codes
        # Attack content latent (first 2D) more strongly as it affects digit identity
        content_noise = torch.randn_like(latent_mu[:, :2]) * epsilon * 2.0
        transform_noise = torch.randn_like(latent_mu[:, 2:]) * epsilon * 0.5
        
        adv_latent_mu = latent_mu.clone()
        adv_latent_mu[:, :2] += content_noise  # Attack content more
        adv_latent_mu[:, 2:] += transform_noise  # Attack transform less
        
        # Reconstruct from adversarial latent
        with torch.no_grad():
            # Sample from adversarial latent distribution
            adv_latent = self.model.reparameterize(adv_latent_mu, latent_logvar)
            
            # Split into content and transform
            adv_content = adv_latent[:, :2]
            adv_transform = adv_latent[:, 2:]
            
            # Decode and apply transform
            adv_clean_recon = self.model.decoder(adv_content)
            adv_final_recon = self.model.apply_affine_transformation(adv_clean_recon, adv_transform)
        
        return adv_final_recon.detach()


def visualize_affine_attack_analysis(original_imgs, attack_imgs, model, device, 
                                   attack_name, epsilon, labels=None):
    """
    Comprehensive visualization for affine autoencoder attacks
    
    Shows 7 rows:
    1. Original images
    2. Original clean reconstructions (decoder only)
    3. Original final reconstructions (after affine)
    4. Adversarial images
    5. Adversarial clean reconstructions 
    6. Adversarial final reconstructions
    7. Latent space comparison (content vs transform)
    """
    model.eval()
    n_samples = min(8, original_imgs.shape[0])
    
    with torch.no_grad():
        # Get original model outputs
        (orig_input, orig_content, orig_transform, _, 
         orig_clean_recon, orig_mu, orig_logvar, orig_final_recon) = model(original_imgs[:n_samples])
        
        # Get adversarial model outputs 
        (adv_input, adv_content, adv_transform, _, 
         adv_clean_recon, adv_mu, adv_logvar, adv_final_recon) = model(attack_imgs[:n_samples])
    
    # Convert to numpy
    orig_np = original_imgs[:n_samples].cpu().numpy()
    attack_np = attack_imgs[:n_samples].cpu().numpy()
    orig_clean_np = orig_clean_recon.cpu().numpy()
    orig_final_np = orig_final_recon.cpu().numpy()
    adv_clean_np = adv_clean_recon.cpu().numpy()
    adv_final_np = adv_final_recon.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(7, n_samples, figsize=(2*n_samples, 14))
    
    row_titles = [
        'Original Images',
        'Original Clean Recon',
        'Original Final Recon', 
        f'{attack_name} Images (ε={epsilon})',
        'Attack Clean Recon',
        'Attack Final Recon',
        'Latent Comparison'
    ]
    
    for i in range(n_samples):
        # Row 1: Original images
        axes[0, i].imshow(orig_np[i, 0], cmap='gray', vmin=0, vmax=1)
        if labels is not None:
            axes[0, i].set_title(f'Label: {labels[i].item()}', fontsize=9)
        axes[0, i].axis('off')
        
        # Row 2: Original clean reconstructions
        axes[1, i].imshow(orig_clean_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        
        # Row 3: Original final reconstructions
        axes[2, i].imshow(orig_final_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        
        # Row 4: Adversarial images
        axes[3, i].imshow(attack_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[3, i].axis('off')
        
        # Row 5: Adversarial clean reconstructions
        axes[4, i].imshow(adv_clean_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[4, i].axis('off')
        
        # Row 6: Adversarial final reconstructions
        axes[5, i].imshow(adv_final_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[5, i].axis('off')
        
        # Row 7: Latent space comparison
        orig_content_vals = orig_content[i].cpu().numpy()
        adv_content_vals = adv_content[i].cpu().numpy()
        orig_transform_vals = orig_transform[i].cpu().numpy()
        adv_transform_vals = adv_transform[i].cpu().numpy()
        
        axes[6, i].text(0.05, 0.9, 'Content:', transform=axes[6, i].transAxes, fontsize=8, weight='bold')
        axes[6, i].text(0.05, 0.8, f'Orig: [{orig_content_vals[0]:.2f}, {orig_content_vals[1]:.2f}]', 
                       transform=axes[6, i].transAxes, fontsize=7)
        axes[6, i].text(0.05, 0.7, f'Adv:  [{adv_content_vals[0]:.2f}, {adv_content_vals[1]:.2f}]', 
                       transform=axes[6, i].transAxes, fontsize=7, color='red')
        
        axes[6, i].text(0.05, 0.5, 'Transform:', transform=axes[6, i].transAxes, fontsize=8, weight='bold')
        axes[6, i].text(0.05, 0.4, f'Δ: {np.linalg.norm(adv_transform_vals - orig_transform_vals):.3f}', 
                       transform=axes[6, i].transAxes, fontsize=7)
        axes[6, i].axis('off')
    
    # Add row titles
    for row, title in enumerate(row_titles):
        axes[row, 0].text(-0.3, 0.5, title, transform=axes[row, 0].transAxes, 
                         rotation=90, va='center', ha='center', fontsize=11, weight='bold')
    
    plt.suptitle(f'🎯 Affine Autoencoder: {attack_name} Attack Analysis (ε={epsilon})', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()


def analyze_attack_effectiveness(original_imgs, attack_imgs, model, device, attack_name, epsilon):
    """Analyze attack effectiveness on affine autoencoder"""
    model.eval()
    
    with torch.no_grad():
        # Original outputs
        (_, orig_content, orig_transform, _, orig_clean, orig_mu, orig_logvar, orig_final) = model(original_imgs)
        
        # Attack outputs
        (_, adv_content, adv_transform, _, adv_clean, adv_mu, adv_logvar, adv_final) = model(attack_imgs)
        
        # Calculate losses
        orig_loss, orig_recon_loss, orig_kl_loss = simplified_affine_kl_loss(
            original_imgs, orig_final, orig_mu, orig_logvar, alpha=1.0, beta=0.008
        )
        
        adv_loss, adv_recon_loss, adv_kl_loss = simplified_affine_kl_loss(
            attack_imgs, adv_final, adv_mu, adv_logvar, alpha=1.0, beta=0.008
        )
    
    # Calculate metrics
    img_mse = F.mse_loss(attack_imgs, original_imgs).item()
    content_shift = F.mse_loss(adv_content, orig_content).item()
    transform_shift = F.mse_loss(adv_transform, orig_transform).item()
    clean_recon_diff = F.mse_loss(adv_clean, orig_clean).item()
    final_recon_diff = F.mse_loss(adv_final, orig_final).item()
    
    print(f"\n📊 {attack_name} Attack Effectiveness Analysis (ε={epsilon}):")
    print(f"  Image Perturbation MSE: {img_mse:.6f}")
    print(f"  Content Latent Shift: {content_shift:.6f}")
    print(f"  Transform Latent Shift: {transform_shift:.6f}")
    print(f"  Clean Reconstruction Diff: {clean_recon_diff:.6f}")
    print(f"  Final Reconstruction Diff: {final_recon_diff:.6f}")
    print(f"  Original Loss: {orig_loss.item():.6f} (Recon: {orig_recon_loss.item():.6f}, KL: {orig_kl_loss.item():.6f})")
    print(f"  Attack Loss: {adv_loss.item():.6f} (Recon: {adv_recon_loss.item():.6f}, KL: {adv_kl_loss.item():.6f})")
    
    # Calculate effectiveness ratios
    content_vs_transform = content_shift / (transform_shift + 1e-8)
    clean_vs_final = clean_recon_diff / (final_recon_diff + 1e-8)
    
    print(f"\n🎯 Attack Characteristics:")
    print(f"  Content vs Transform Impact Ratio: {content_vs_transform:.3f}")
    print(f"  Clean vs Final Reconstruction Ratio: {clean_vs_final:.3f}")
    
    if content_vs_transform > 2.0:
        print("  → Attack primarily affects CONTENT (digit identity)")
    elif content_vs_transform < 0.5:
        print("  → Attack primarily affects TRANSFORM (affine parameters)")
    else:
        print("  → Attack affects both CONTENT and TRANSFORM")
    
    return {
        'img_mse': img_mse,
        'content_shift': content_shift,
        'transform_shift': transform_shift,
        'clean_recon_diff': clean_recon_diff,
        'final_recon_diff': final_recon_diff,
        'orig_loss': orig_loss.item(),
        'adv_loss': adv_loss.item(),
        'content_vs_transform_ratio': content_vs_transform
    }


def run_comprehensive_affine_attacks(model, test_loader, device, config):
    """Run all attack methods on the affine autoencoder"""
    print("🚀 Running Comprehensive Affine Autoencoder Attack Analysis")
    
    # Get test samples randomly
    test_iter = iter(test_loader)
    test_images, test_labels = next(test_iter)
    
    batch_size = test_images.size(0)
    n_samples = config.get('n_attack_samples', 8)
    random_indices = random.sample(range(batch_size), min(n_samples, batch_size))
    
    samples = test_images[random_indices].to(device)
    labels = test_labels[random_indices]
    
    print(f"Selected {len(samples)} random samples for attack analysis")
    
    # Initialize attacker
    attacker = AffineAdversarialAttacks(model, device)
    
    results = {}
    
    # Test different epsilon values
    epsilons = config.get('attack_epsilons', [0.05, 0.1, 0.15, 0.2])
    
    for epsilon in epsilons:
        print(f"\n{'='*60}")
        print(f"Testing attacks with ε = {epsilon}")
        print(f"{'='*60}")
        
        # FGSM Attack
        print(f"\n🎯 Running FGSM Attack (ε={epsilon})")
        fgsm_imgs = attacker.fgsm_attack(samples, labels, epsilon)
        visualize_affine_attack_analysis(samples, fgsm_imgs, model, device, 
                                       "FGSM", epsilon, labels)
        fgsm_results = analyze_attack_effectiveness(samples, fgsm_imgs, model, device, 
                                                  "FGSM", epsilon)
        
        # PGD Attack
        print(f"\n🎯 Running PGD Attack (ε={epsilon})")
        pgd_imgs = attacker.pgd_attack(samples, labels, epsilon)
        visualize_affine_attack_analysis(samples, pgd_imgs, model, device, 
                                       "PGD", epsilon, labels)
        pgd_results = analyze_attack_effectiveness(samples, pgd_imgs, model, device, 
                                                 "PGD", epsilon)
        
        # Latent Space Attack
        print(f"\n🎯 Running Latent Space Attack (ε={epsilon})")
        latent_imgs = attacker.latent_space_attack(samples, labels, epsilon)
        visualize_affine_attack_analysis(samples, latent_imgs, model, device, 
                                       "Latent", epsilon, labels)
        latent_results = analyze_attack_effectiveness(samples, latent_imgs, model, device, 
                                                    "Latent", epsilon)
        
        # Store results
        results[f'epsilon_{epsilon}'] = {
            'fgsm': fgsm_results,
            'pgd': pgd_results,
            'latent': latent_results
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"affine_attack_analysis_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("🏁 Comprehensive attack analysis completed!")
    
    return results


def compare_attack_methods(results):
    """Compare effectiveness of different attack methods"""
    print("\n📈 Attack Methods Comparison")
    print("="*50)
    
    methods = ['fgsm', 'pgd', 'latent']
    epsilons = [key for key in results.keys() if key.startswith('epsilon_')]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for method in methods:
        eps_vals = []
        content_shifts = []
        transform_shifts = []
        final_diffs = []
        
        for eps_key in epsilons:
            eps_val = float(eps_key.split('_')[1])
            eps_vals.append(eps_val)
            content_shifts.append(results[eps_key][method]['content_shift'])
            transform_shifts.append(results[eps_key][method]['transform_shift'])
            final_diffs.append(results[eps_key][method]['final_recon_diff'])
        
        # Content shift plot
        axes[0, 0].plot(eps_vals, content_shifts, marker='o', label=method.upper())
        axes[0, 0].set_title('Content Latent Shift vs Epsilon')
        axes[0, 0].set_xlabel('Epsilon')
        axes[0, 0].set_ylabel('Content Shift (MSE)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Transform shift plot
        axes[0, 1].plot(eps_vals, transform_shifts, marker='s', label=method.upper())
        axes[0, 1].set_title('Transform Latent Shift vs Epsilon')
        axes[0, 1].set_xlabel('Epsilon')
        axes[0, 1].set_ylabel('Transform Shift (MSE)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Final reconstruction difference
        axes[1, 0].plot(eps_vals, final_diffs, marker='^', label=method.upper())
        axes[1, 0].set_title('Final Reconstruction Difference vs Epsilon')
        axes[1, 0].set_xlabel('Epsilon')
        axes[1, 0].set_ylabel('Reconstruction Diff (MSE)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Content vs Transform ratio comparison
    for method in methods:
        eps_vals = []
        ratios = []
        
        for eps_key in epsilons:
            eps_val = float(eps_key.split('_')[1])
            eps_vals.append(eps_val)
            ratios.append(results[eps_key][method]['content_vs_transform_ratio'])
        
        axes[1, 1].plot(eps_vals, ratios, marker='d', label=method.upper())
    
    axes[1, 1].set_title('Content vs Transform Impact Ratio')
    axes[1, 1].set_xlabel('Epsilon')  
    axes[1, 1].set_ylabel('Content/Transform Ratio')
    axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal Impact')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.suptitle('🎯 Affine Autoencoder: Attack Methods Comparison', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()
