# attack_analysis.py
"""
Comprehensive adversarial attack analysis module for VAE models.

This module contains all functions for:
- Adversarial attack visualization and analysis
- Attack effectiveness comparison
- Attack efficiency scaling analysis  
- Latent space clustering analysis under attacks
- Statistical analysis and visualization functions
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression
from adversarial_vae_attack import AdversarialAttacks


def visualize_attack_analysis(original_imgs, original_recons, attack_imgs, attack_recons, 
                            attack_name, epsilon, labels=None):
    """
    Comprehensive visualization of adversarial attack analysis
    
    Shows 6 rows:
    1. Original images
    2. Original reconstructions
    3. Adversarial images
    4. Adversarial reconstructions
    5. Image differences (original vs adversarial)
    6. Reconstruction differences
    """
    n_samples = original_imgs.shape[0]
    
    # Convert to numpy for visualization (detach to handle gradients)
    orig_np = original_imgs.detach().cpu().numpy()
    orig_recon_np = original_recons.detach().cpu().numpy()
    attack_np = attack_imgs.detach().cpu().numpy()
    attack_recon_np = attack_recons.detach().cpu().numpy()
    
    # Calculate differences
    img_diff = np.abs(attack_np - orig_np)
    recon_diff = np.abs(attack_recon_np - orig_recon_np)
    
    # Create figure
    fig, axes = plt.subplots(6, n_samples, figsize=(2*n_samples, 12))
    
    row_titles = [
        'Original Images',
        'Original Reconstructions', 
        f'{attack_name} (ε={epsilon})',
        'Attack Reconstructions',
        'Image Differences',
        'Reconstruction Differences'
    ]
    
    for i in range(n_samples):
        # Row 1: Original images
        axes[0, i].imshow(orig_np[i, 0], cmap='gray', vmin=0, vmax=1)
        if labels is not None:
            axes[0, i].set_title(f'Label: {labels[i].item()}')
        axes[0, i].axis('off')
        
        # Row 2: Original reconstructions
        axes[1, i].imshow(orig_recon_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        
        # Row 3: Adversarial images
        axes[2, i].imshow(attack_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        
        # Row 4: Adversarial reconstructions
        axes[3, i].imshow(attack_recon_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[3, i].axis('off')
        
        # Row 5: Image differences (use different colormap for differences)
        im1 = axes[4, i].imshow(img_diff[i, 0], cmap='hot', vmin=0)
        axes[4, i].axis('off')
        
        # Row 6: Reconstruction differences
        im2 = axes[5, i].imshow(recon_diff[i, 0], cmap='hot', vmin=0)
        axes[5, i].axis('off')
    
    # Add row titles
    for row, title in enumerate(row_titles):
        axes[row, 0].text(-0.5, 0.5, title, transform=axes[row, 0].transAxes, 
                         rotation=90, va='center', ha='center', fontsize=12, weight='bold')
    
    plt.suptitle(f'{attack_name} Attack Analysis (ε={epsilon})', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()
    
    # Calculate and display metrics
    img_mse = F.mse_loss(attack_imgs.detach(), original_imgs.detach()).item()
    recon_mse = F.mse_loss(attack_recons.detach(), original_recons.detach()).item()
    attack_success = (img_mse > 0.001)  # Simple success metric
    
    print(f"📊 {attack_name} Attack Metrics (ε={epsilon}):")
    print(f"   Image MSE: {img_mse:.6f}")
    print(f"   Reconstruction MSE: {recon_mse:.6f}")
    print(f"   Max image perturbation: {img_diff.max():.6f}")
    print(f"   Max reconstruction change: {recon_diff.max():.6f}")
    print(f"   Attack success: {'✅' if attack_success else '❌'}")
    print("-" * 50)


def custom_vae_attack(model, data, epsilon, num_iter=10, alpha=None, target_latent=None):
    """
    Custom iterative attack targeting both reconstruction and latent space
    """
    if alpha is None:
        alpha = epsilon / num_iter
    
    model.eval()
    
    # Initialize perturbation
    delta = torch.zeros_like(data).uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    
    for i in range(num_iter):
        # Forward pass with perturbation
        perturbed_data = data + delta
        recon_data, mu, logvar = model(perturbed_data)
        
        # Combined loss: reconstruction + KL divergence + latent targeting
        recon_loss = F.binary_cross_entropy(recon_data, data, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = recon_loss + 0.1 * kl_loss
        
        # Add latent targeting if specified
        if target_latent is not None:
            latent_loss = F.mse_loss(mu, target_latent)
            loss += latent_loss
        
        # Calculate gradients
        loss.backward()
        
        # Update perturbation
        delta.data = delta.data + alpha * delta.grad.data.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = torch.clamp(data + delta.data, 0, 1) - data
        
        # Zero gradients
        delta.grad.zero_()
    
    return data + delta


def compare_attack_effectiveness(model, data, epsilon=0.1):
    """
    Compare different attack methods on the same data
    """
    attacks = AdversarialAttacks()
    results = {}
    
    # Get original reconstructions
    with torch.no_grad():
        orig_recons, orig_mu, orig_logvar = model(data)
    
    # FGSM Attack
    fgsm_imgs = attacks.fgsm_attack(model, data, data, epsilon)
    with torch.no_grad():
        fgsm_recons, fgsm_mu, fgsm_logvar = model(fgsm_imgs)
    
    # PGD Attack
    pgd_imgs = attacks.pgd_attack(model, data, data, epsilon, alpha=epsilon/10, num_iter=20)
    with torch.no_grad():
        pgd_recons, pgd_mu, pgd_logvar = model(pgd_imgs)
    
    # Custom Attack
    custom_imgs = custom_vae_attack(model, data, epsilon, num_iter=15)
    with torch.no_grad():
        custom_recons, custom_mu, custom_logvar = model(custom_imgs)
    
    # Calculate metrics for each attack
    attacks_data = {
        'FGSM': (fgsm_imgs, fgsm_recons, fgsm_mu),
        'PGD': (pgd_imgs, pgd_recons, pgd_mu),
        'Custom': (custom_imgs, custom_recons, custom_mu)
    }
    
    for attack_name, (attack_imgs, attack_recons, attack_mu) in attacks_data.items():
        # Image perturbation metrics
        img_mse = F.mse_loss(attack_imgs, data).item()
        img_linf = (attack_imgs - data).abs().max().item()
        
        # Reconstruction metrics
        recon_mse = F.mse_loss(attack_recons, orig_recons).item()
        
        # Latent space metrics
        latent_mse = F.mse_loss(attack_mu, orig_mu).item()
        
        results[attack_name] = {
            'img_mse': img_mse,
            'img_linf': img_linf,
            'recon_mse': recon_mse,
            'latent_mse': latent_mse
        }
    
    return results


def analyze_attack_success_rates(model, test_loader, epsilons, device, max_batches=10):
    """
    Analyze attack success rates across different epsilon values and digit classes
    """
    attacks = AdversarialAttacks()
    results = {eps: {'fgsm': [], 'pgd': [], 'by_digit': {i: {'fgsm': [], 'pgd': []} for i in range(10)}} 
               for eps in epsilons}
    
    model.eval()
    total_samples = 0
    
    for batch_idx, (data, labels) in enumerate(test_loader):
        if batch_idx >= max_batches:
            break
            
        data = data.to(device)
        labels = labels.to(device)
        
        # Get original reconstructions
        with torch.no_grad():
            orig_recons, _, _ = model(data)
        
        for eps in epsilons:
            # FGSM attack (needs gradients)
            fgsm_imgs = attacks.fgsm_attack(model, data, data, eps)
            with torch.no_grad():
                fgsm_recons, _, _ = model(fgsm_imgs)
            
            # PGD attack (needs gradients)
            pgd_imgs = attacks.pgd_attack(model, data, data, eps, alpha=eps/10, num_iter=10)
            with torch.no_grad():
                pgd_recons, _, _ = model(pgd_imgs)
                
                # Calculate success (significant change in reconstruction)
                fgsm_success = (F.mse_loss(fgsm_recons, orig_recons, reduction='none').mean(dim=[1,2,3]) > 0.01).float()
                pgd_success = (F.mse_loss(pgd_recons, orig_recons, reduction='none').mean(dim=[1,2,3]) > 0.01).float()
                
                results[eps]['fgsm'].extend(fgsm_success.cpu().numpy())
                results[eps]['pgd'].extend(pgd_success.cpu().numpy())
                
                # Track by digit class
                for digit in range(10):
                    digit_mask = (labels == digit)
                    if digit_mask.sum() > 0:
                        results[eps]['by_digit'][digit]['fgsm'].extend(fgsm_success[digit_mask].cpu().numpy())
                        results[eps]['by_digit'][digit]['pgd'].extend(pgd_success[digit_mask].cpu().numpy())
            
            total_samples += data.size(0)
    
    return results, total_samples


def comprehensive_attack_efficiency(model, test_loader, epsilon_range, device, max_samples=1000):
    """
    Comprehensive attack efficiency analysis across the full test dataset
    """
    attacks = AdversarialAttacks()
    model.eval()
    
    # Initialize results storage
    results = {
        'epsilons': epsilon_range,
        'fgsm': {'success_rate': [], 'recon_mse': [], 'latent_shift': [], 'perceptual_distance': []},
        'pgd': {'success_rate': [], 'recon_mse': [], 'latent_shift': [], 'perceptual_distance': []},
        'custom': {'success_rate': [], 'recon_mse': [], 'latent_shift': [], 'perceptual_distance': []},
        'sample_count': 0
    }
    
    # Collect test samples
    print("📊 Collecting test samples...")
    all_data, all_labels = [], []
    sample_count = 0
    
    for data, labels in test_loader:
        if sample_count >= max_samples:
            break
        
        batch_size = min(data.size(0), max_samples - sample_count)
        all_data.append(data[:batch_size].to(device))
        all_labels.append(labels[:batch_size].to(device))
        sample_count += batch_size
    
    # Concatenate all data
    test_data = torch.cat(all_data, dim=0)
    test_labels = torch.cat(all_labels, dim=0)
    results['sample_count'] = test_data.size(0)
    
    print(f"Testing on {results['sample_count']} samples")
    
    # Get original reconstructions and latent representations
    print("Getting original reconstructions...")
    with torch.no_grad():
        orig_recons, orig_mu, orig_logvar = model(test_data)
    
    # Test each epsilon value
    for i, eps in enumerate(epsilon_range):
        print(f"\n🔥 Testing epsilon = {eps:.3f} ({i+1}/{len(epsilon_range)})")
        
        # FGSM Attack
        print("  Running FGSM...")
        fgsm_imgs = attacks.fgsm_attack(model, test_data, test_data, eps)
        with torch.no_grad():
            fgsm_recons, fgsm_mu, fgsm_logvar = model(fgsm_imgs)
        
        # PGD Attack
        print("  Running PGD...")
        pgd_imgs = attacks.pgd_attack(model, test_data, test_data, eps, alpha=eps/10, num_iter=10)
        with torch.no_grad():
            pgd_recons, pgd_mu, pgd_logvar = model(pgd_imgs)
        
        # Custom Attack
        print("  Running Custom...")
        custom_imgs = custom_vae_attack(model, test_data, eps, num_iter=10)
        with torch.no_grad():
            custom_recons, custom_mu, custom_logvar = model(custom_imgs)
        
        # Calculate metrics for each attack
        attack_data = {
            'fgsm': (fgsm_imgs, fgsm_recons, fgsm_mu),
            'pgd': (pgd_imgs, pgd_recons, pgd_mu),
            'custom': (custom_imgs, custom_recons, custom_mu)
        }
        
        for attack_name, (attack_imgs, attack_recons, attack_mu) in attack_data.items():
            # Success rate (based on reconstruction MSE threshold)
            recon_mse_per_sample = F.mse_loss(attack_recons, orig_recons, reduction='none').mean(dim=[1,2,3])
            success_mask = recon_mse_per_sample > 0.01  # Threshold for "successful" attack
            success_rate = success_mask.float().mean().item()
            
            # Average reconstruction MSE
            avg_recon_mse = recon_mse_per_sample.mean().item()
            
            # Latent space shift magnitude
            latent_shift = F.mse_loss(attack_mu, orig_mu, reduction='none').mean(dim=1).mean().item()
            
            # Perceptual distance (L2 norm of image difference)
            perceptual_dist = F.mse_loss(attack_imgs, test_data, reduction='none').mean(dim=[1,2,3]).mean().item()
            
            # Store results
            results[attack_name]['success_rate'].append(success_rate)
            results[attack_name]['recon_mse'].append(avg_recon_mse)
            results[attack_name]['latent_shift'].append(latent_shift)
            results[attack_name]['perceptual_distance'].append(perceptual_dist)
    
    return results


def plot_attack_efficiency_scaling(results):
    """
    Create comprehensive plots showing how attack efficiency scales with epsilon
    """
    epsilons = results['epsilons']
    
    # Set up the plot with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Attack Efficiency Scaling with Epsilon Parameter', fontsize=16, fontweight='bold')
    
    # Colors and markers for different attacks
    colors = {'fgsm': '#FF6B6B', 'pgd': '#4ECDC4', 'custom': '#45B7D1'}
    markers = {'fgsm': 'o', 'pgd': 's', 'custom': '^'}
    labels = {'fgsm': 'FGSM', 'pgd': 'PGD', 'custom': 'Custom'}
    
    # Plot 1: Success Rate vs Epsilon
    ax1 = axes[0, 0]
    for attack_name in ['fgsm', 'pgd', 'custom']:
        success_rates = np.array(results[attack_name]['success_rate']) * 100
        ax1.plot(epsilons, success_rates, marker=markers[attack_name], 
                color=colors[attack_name], label=labels[attack_name], 
                linewidth=2, markersize=6, alpha=0.8)
    
    ax1.set_xlabel('Epsilon (Perturbation Strength)')
    ax1.set_ylabel('Attack Success Rate (%)')
    ax1.set_title('Attack Success Rate vs Epsilon')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Plot 2: Reconstruction MSE vs Epsilon
    ax2 = axes[0, 1]
    for attack_name in ['fgsm', 'pgd', 'custom']:
        recon_mse = results[attack_name]['recon_mse']
        ax2.semilogy(epsilons, recon_mse, marker=markers[attack_name], 
                    color=colors[attack_name], label=labels[attack_name], 
                    linewidth=2, markersize=6, alpha=0.8)
    
    ax2.set_xlabel('Epsilon (Perturbation Strength)')
    ax2.set_ylabel('Reconstruction MSE (log scale)')
    ax2.set_title('Reconstruction Disruption vs Epsilon')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Latent Space Shift vs Epsilon
    ax3 = axes[1, 0]
    for attack_name in ['fgsm', 'pgd', 'custom']:
        latent_shift = results[attack_name]['latent_shift']
        ax3.plot(epsilons, latent_shift, marker=markers[attack_name], 
                color=colors[attack_name], label=labels[attack_name], 
                linewidth=2, markersize=6, alpha=0.8)
    
    ax3.set_xlabel('Epsilon (Perturbation Strength)')
    ax3.set_ylabel('Latent Space Shift (MSE)')
    ax3.set_title('Latent Space Disruption vs Epsilon')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Perceptual Distance vs Epsilon
    ax4 = axes[1, 1]
    for attack_name in ['fgsm', 'pgd', 'custom']:
        perceptual_dist = results[attack_name]['perceptual_distance']
        ax4.plot(epsilons, perceptual_dist, marker=markers[attack_name], 
                color=colors[attack_name], label=labels[attack_name], 
                linewidth=2, markersize=6, alpha=0.8)
    
    ax4.set_xlabel('Epsilon (Perturbation Strength)')
    ax4.set_ylabel('Perceptual Distance (Image MSE)')
    ax4.set_title('Input Perturbation vs Epsilon')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_simple_efficiency_summary(results):
    """
    Create a simple but effective plot showing overall attack efficiency vs epsilon
    """
    epsilons = results['epsilons']
    
    plt.figure(figsize=(12, 8))
    
    # Plot success rates with enhanced styling
    colors = {'fgsm': '#FF6B6B', 'pgd': '#4ECDC4', 'custom': '#45B7D1'}
    labels = {'fgsm': 'FGSM', 'pgd': 'PGD', 'custom': 'Custom Iterative'}
    
    for attack_name in ['fgsm', 'pgd', 'custom']:
        success_rates = np.array(results[attack_name]['success_rate']) * 100
        plt.plot(epsilons, success_rates, 'o-', color=colors[attack_name], 
                label=labels[attack_name], linewidth=3, markersize=8, alpha=0.9)
        
        # Add fill under curve for visual appeal
        plt.fill_between(epsilons, success_rates, alpha=0.2, color=colors[attack_name])
    
    # Styling
    plt.xlabel('Epsilon (Perturbation Strength)', fontsize=14, fontweight='bold')
    plt.ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Adversarial Attack Efficiency vs Perturbation Strength\n' + 
              f'Tested on {results["sample_count"]} MNIST samples', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(epsilons[0], epsilons[-1])
    plt.ylim(0, 105)
    
    # Add some key annotations
    for attack_name in ['pgd']:  # Annotate the best performing attack
        success_rates = np.array(results[attack_name]['success_rate']) * 100
        max_idx = np.argmax(success_rates)
        max_eps = epsilons[max_idx]
        max_rate = success_rates[max_idx]
        
        plt.annotate(f'Peak: {max_rate:.1f}% at ε={max_eps:.2f}', 
                    xy=(max_eps, max_rate), xytext=(max_eps-0.05, max_rate+10),
                    arrowprops=dict(arrowstyle='->', color=colors[attack_name], lw=2),
                    fontsize=11, fontweight='bold', color=colors[attack_name],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


# Latent Space Clustering Analysis Functions

def sample_digits_by_class(test_loader, device, samples_per_class=15, max_batches=20):
    """
    Sample random digits from each class (0-9) for latent space analysis
    """
    digit_samples = {i: {'images': [], 'labels': [], 'indices': []} for i in range(10)}
    batch_count = 0
    
    print("🔍 Collecting digit samples by class...")
    
    for batch_idx, (data, labels) in enumerate(test_loader):
        if batch_count >= max_batches:
            break
            
        data = data.to(device)
        labels = labels.to(device)
        
        # Collect samples for each digit class
        for digit in range(10):
            if len(digit_samples[digit]['images']) >= samples_per_class:
                continue
                
            digit_mask = (labels == digit)
            digit_indices = torch.where(digit_mask)[0]
            
            if len(digit_indices) > 0:
                # Take samples needed to reach target
                needed = samples_per_class - len(digit_samples[digit]['images'])
                take_count = min(needed, len(digit_indices))
                
                selected_indices = digit_indices[:take_count]
                digit_samples[digit]['images'].append(data[selected_indices])
                digit_samples[digit]['labels'].append(labels[selected_indices])
                digit_samples[digit]['indices'].extend([batch_idx * test_loader.batch_size + idx.item() for idx in selected_indices])
        
        batch_count += 1
        
        # Check if we have enough samples for all digits
        if all(len(digit_samples[d]['images']) >= samples_per_class for d in range(10)):
            break
    
    # Concatenate collected samples
    final_samples = {}
    for digit in range(10):
        if digit_samples[digit]['images']:
            final_samples[digit] = {
                'images': torch.cat(digit_samples[digit]['images'], dim=0)[:samples_per_class],
                'labels': torch.cat(digit_samples[digit]['labels'], dim=0)[:samples_per_class],
                'indices': digit_samples[digit]['indices'][:samples_per_class]
            }
            print(f"  Digit {digit}: {len(final_samples[digit]['images'])} samples")
        else:
            print(f"  Digit {digit}: No samples found!")
    
    return final_samples


def analyze_latent_clustering_under_attack(model, digit_samples, attack_method='fgsm', epsilon=0.1):
    """
    Analyze how adversarial attacks affect latent space clustering of digit classes
    """
    attacks = AdversarialAttacks()
    model.eval()
    
    results = {
        'original': {'latent': {}, 'images': {}},
        'adversarial': {'latent': {}, 'images': {}},
        'attack_info': {'method': attack_method, 'epsilon': epsilon}
    }
    
    print(f"🎯 Analyzing latent clustering under {attack_method.upper()} attack (ε={epsilon})")
    
    for digit in range(10):
        if digit not in digit_samples:
            continue
            
        images = digit_samples[digit]['images']
        print(f"  Processing digit {digit}: {len(images)} samples")
        
        # Get original latent representations
        with torch.no_grad():
            _, orig_mu, orig_logvar = model(images)
        
        # Apply adversarial attack
        if attack_method == 'fgsm':
            adv_images = attacks.fgsm_attack(model, images, images, epsilon)
        elif attack_method == 'pgd':
            adv_images = attacks.pgd_attack(model, images, images, epsilon, alpha=epsilon/10, num_iter=10)
        elif attack_method == 'custom':
            adv_images = custom_vae_attack(model, images, epsilon, num_iter=10)
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
        
        # Get adversarial latent representations
        with torch.no_grad():
            _, adv_mu, adv_logvar = model(adv_images)
        
        # Store results
        results['original']['latent'][digit] = orig_mu.cpu().numpy()
        results['original']['images'][digit] = images.detach().cpu().numpy()
        results['adversarial']['latent'][digit] = adv_mu.cpu().numpy()
        results['adversarial']['images'][digit] = adv_images.detach().cpu().numpy()
    
    return results


def plot_latent_clustering_comparison(clustering_results):
    """
    Create comprehensive plots showing original vs adversarial latent clustering
    """
    attack_info = clustering_results['attack_info']
    
    # Set up the plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Latent Space Clustering: Original vs {attack_info["method"].upper()} Attack (ε={attack_info["epsilon"]})', 
                 fontsize=16, fontweight='bold')
    
    # Define colors for each digit
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot 1: Original latent space
    ax1 = axes[0]
    for digit in range(10):
        if digit in clustering_results['original']['latent']:
            latent_points = clustering_results['original']['latent'][digit]
            ax1.scatter(latent_points[:, 0], latent_points[:, 1], 
                       c=[colors[digit]], label=f'Digit {digit}', alpha=0.7, s=50)
    
    ax1.set_xlabel('Latent Dimension 1')
    ax1.set_ylabel('Latent Dimension 2')
    ax1.set_title('Original Latent Space')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Adversarial latent space
    ax2 = axes[1]
    for digit in range(10):
        if digit in clustering_results['adversarial']['latent']:
            latent_points = clustering_results['adversarial']['latent'][digit]
            ax2.scatter(latent_points[:, 0], latent_points[:, 1], 
                       c=[colors[digit]], label=f'Digit {digit}', alpha=0.7, s=50)
    
    ax2.set_xlabel('Latent Dimension 1')
    ax2.set_ylabel('Latent Dimension 2')
    ax2.set_title(f'After {attack_info["method"].upper()} Attack')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Movement vectors (before -> after)
    ax3 = axes[2]
    for digit in range(10):
        if digit in clustering_results['original']['latent'] and digit in clustering_results['adversarial']['latent']:
            orig_points = clustering_results['original']['latent'][digit]
            adv_points = clustering_results['adversarial']['latent'][digit]
            
            # Plot original points
            ax3.scatter(orig_points[:, 0], orig_points[:, 1], 
                       c=[colors[digit]], alpha=0.4, s=30, marker='o')
            
            # Plot adversarial points
            ax3.scatter(adv_points[:, 0], adv_points[:, 1], 
                       c=[colors[digit]], alpha=0.8, s=30, marker='x')
            
            # Draw movement vectors
            for i in range(len(orig_points)):
                ax3.arrow(orig_points[i, 0], orig_points[i, 1],
                         adv_points[i, 0] - orig_points[i, 0],
                         adv_points[i, 1] - orig_points[i, 1],
                         head_width=0.02, head_length=0.02, 
                         fc=colors[digit], ec=colors[digit], alpha=0.6, length_includes_head=True)
    
    ax3.set_xlabel('Latent Dimension 1')
    ax3.set_ylabel('Latent Dimension 2')
    ax3.set_title('Movement Vectors (○ original → × adversarial)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display clustering metrics
    print("\n" + "="*80)
    print("LATENT SPACE CLUSTERING ANALYSIS")
    print("="*80)
    
    # Calculate centroid movements
    print(f"\nCentroid Movements for {attack_info['method'].upper()} Attack (ε={attack_info['epsilon']}):")
    print("-" * 60)
    
    total_movement = 0
    digit_movements = []
    
    for digit in range(10):
        if digit in clustering_results['original']['latent'] and digit in clustering_results['adversarial']['latent']:
            orig_centroid = np.mean(clustering_results['original']['latent'][digit], axis=0)
            adv_centroid = np.mean(clustering_results['adversarial']['latent'][digit], axis=0)
            movement = np.linalg.norm(adv_centroid - orig_centroid)
            
            print(f"  Digit {digit}: Centroid moved {movement:.4f} units")
            digit_movements.append(movement)
            total_movement += movement
    
    if digit_movements:
        print(f"\nOverall Statistics:")
        print(f"  Average centroid movement: {np.mean(digit_movements):.4f}")
        print(f"  Max centroid movement: {np.max(digit_movements):.4f}")
        print(f"  Min centroid movement: {np.min(digit_movements):.4f}")
        print(f"  Std deviation: {np.std(digit_movements):.4f}")
    
    # Calculate intra-class scatter changes
    print(f"\nIntra-class Scatter Changes:")
    print("-" * 40)
    
    for digit in range(10):
        if digit in clustering_results['original']['latent'] and digit in clustering_results['adversarial']['latent']:
            orig_points = clustering_results['original']['latent'][digit]
            adv_points = clustering_results['adversarial']['latent'][digit]
            
            orig_centroid = np.mean(orig_points, axis=0)
            adv_centroid = np.mean(adv_points, axis=0)
            
            orig_scatter = np.mean(np.linalg.norm(orig_points - orig_centroid, axis=1))
            adv_scatter = np.mean(np.linalg.norm(adv_points - adv_centroid, axis=1))
            
            scatter_change = ((adv_scatter - orig_scatter) / orig_scatter) * 100
            
            print(f"  Digit {digit}: Scatter changed by {scatter_change:+.1f}% "
                  f"({orig_scatter:.4f} → {adv_scatter:.4f})")


def compare_latent_disruption_across_attacks(clustering_analyses):
    """
    Create a comprehensive comparison of how different attacks affect latent space clustering
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparative Analysis: Latent Space Disruption Across Attack Methods', 
                 fontsize=16, fontweight='bold')
    
    attack_methods = list(clustering_analyses.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot 1: Centroid movement comparison
    ax1 = axes[0, 0]
    centroid_movements = {method: [] for method in attack_methods}
    
    for method in attack_methods:
        results = clustering_analyses[method]
        for digit in range(10):
            if (digit in results['original']['latent'] and 
                digit in results['adversarial']['latent']):
                orig_centroid = np.mean(results['original']['latent'][digit], axis=0)
                adv_centroid = np.mean(results['adversarial']['latent'][digit], axis=0)
                movement = np.linalg.norm(adv_centroid - orig_centroid)
                centroid_movements[method].append(movement)
    
    # Box plot of centroid movements
    movement_data = [centroid_movements[method] for method in attack_methods]
    bp = ax1.boxplot(movement_data, labels=[m.upper() for m in attack_methods], patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4', '#45B7D1']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Centroid Movement Distance')
    ax1.set_title('Centroid Movement by Attack Method')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter increase/decrease
    ax2 = axes[0, 1]
    scatter_changes = {method: [] for method in attack_methods}
    
    for method in attack_methods:
        results = clustering_analyses[method]
        for digit in range(10):
            if (digit in results['original']['latent'] and 
                digit in results['adversarial']['latent']):
                orig_points = results['original']['latent'][digit]
                adv_points = results['adversarial']['latent'][digit]
                
                orig_centroid = np.mean(orig_points, axis=0)
                adv_centroid = np.mean(adv_points, axis=0)
                
                orig_scatter = np.mean(np.linalg.norm(orig_points - orig_centroid, axis=1))
                adv_scatter = np.mean(np.linalg.norm(adv_points - adv_centroid, axis=1))
                
                scatter_change = ((adv_scatter - orig_scatter) / orig_scatter) * 100
                scatter_changes[method].append(scatter_change)
    
    # Bar plot of average scatter changes
    avg_changes = [np.mean(scatter_changes[method]) for method in attack_methods]
    bars = ax2.bar([m.upper() for m in attack_methods], avg_changes, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    
    ax2.set_ylabel('Average Scatter Change (%)')
    ax2.set_title('Intra-class Scatter Changes')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_changes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Plot 3: Attack strength vs movement correlation
    ax3 = axes[1, 0]
    for i, method in enumerate(attack_methods):
        movements = centroid_movements[method]
        digit_labels = list(range(len(movements)))
        ax3.scatter(digit_labels, movements, label=method.upper(), 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i], alpha=0.7, s=60)
    
    ax3.set_xlabel('Digit Class')
    ax3.set_ylabel('Centroid Movement')
    ax3.set_title('Movement by Digit Class')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(10))
    
    # Plot 4: Latent space coverage comparison
    ax4 = axes[1, 1]
    coverage_stats = []
    
    for method in attack_methods:
        results = clustering_analyses[method]
        all_orig_points = []
        all_adv_points = []
        
        for digit in range(10):
            if (digit in results['original']['latent'] and 
                digit in results['adversarial']['latent']):
                all_orig_points.extend(results['original']['latent'][digit])
                all_adv_points.extend(results['adversarial']['latent'][digit])
        
        if all_orig_points and all_adv_points:
            orig_points = np.array(all_orig_points)
            adv_points = np.array(all_adv_points)
            
            # Calculate coverage area (using convex hull area approximation)
            orig_range_x = np.ptp(orig_points[:, 0])  # peak-to-peak
            orig_range_y = np.ptp(orig_points[:, 1])
            orig_coverage = orig_range_x * orig_range_y
            
            adv_range_x = np.ptp(adv_points[:, 0])
            adv_range_y = np.ptp(adv_points[:, 1])
            adv_coverage = adv_range_x * adv_range_y
            
            coverage_change = ((adv_coverage - orig_coverage) / orig_coverage) * 100
            coverage_stats.append(coverage_change)
        else:
            coverage_stats.append(0)
    
    bars = ax4.bar([m.upper() for m in attack_methods], coverage_stats,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    
    ax4.set_ylabel('Coverage Area Change (%)')
    ax4.set_title('Latent Space Coverage Changes')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, value in zip(bars, coverage_stats):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.show()


def save_analysis_results(results, model_timestamp, analysis_type='general'):
    """
    Save analysis results to JSON file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_data = {
        'timestamp': timestamp,
        'model_timestamp': model_timestamp,
        'analysis_type': analysis_type,
        'results': results
    }
    
    filename = f"{analysis_type}_analysis_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    print(f"\n💾 Analysis results saved to: {filename}")
    return filename
