"""
Affine Adversarial Attack Analysis Functions

This module contains utility functions extracted from the adversarial attack analysis notebook.
These functions provide comprehensive analysis capabilities for the structured affine autoencoder
including non-digit image generation, attack evaluation, and statistical analysis.

Functions moved from notebook include:
- random_sample_attack_data: Random sampling utility for attack samples
- generate_non_digit_images: Generate completely non-digit-like test images
- analyze_non_digit_reconstructions: Analyze how the model reconstructs non-digit images
- find_optimal_inputs_for_coordinates: Inverse engineering for specific latent coordinates
- explore_coordinate_space: Comprehensive latent space exploration
- targeted_content_attack: Attack targeting content latent dimensions
- targeted_transform_attack: Attack targeting transform latent dimensions
- evaluate_adversarial_training_benefit: Simulate adversarial training effects
- create_synthetic_attack_targets: Generate synthetic adversarial examples
- statistical_attack_analysis: Statistical analysis of attack effectiveness
- generate_attack_report: Generate comprehensive attack analysis report
- manual_reparameterize: Helper for VAE reparameterization
- access_inner_autoencoder_components: Helper for accessing wrapped model components
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

# Import required modules (these should be available in the environment)
try:
    from affine_autoencoder_shared import simplified_affine_kl_loss
    import affine_adversarial_attacks
    from affine_adversarial_attacks import AffineAdversarialAttacks
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Make sure affine_autoencoder_shared and affine_adversarial_attacks are available")


def random_sample_attack_data(attack_samples_cpu, attack_labels, n_random_samples=None, seed=None):
    """
    Randomly sample from attack_samples_cpu and their matching labels.
    
    Args:
        attack_samples_cpu: Tensor of attack samples on CPU
        attack_labels: Tensor of corresponding labels
        n_random_samples: Number of random samples to select (default: use all)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Tuple of (sampled_attack_samples_cpu, sampled_attack_labels)
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    total_samples = len(attack_samples_cpu)
    
    # Default to using all samples if n_random_samples not specified
    if n_random_samples is None:
        n_random_samples = total_samples
    
    # Ensure we don't sample more than available
    n_random_samples = min(n_random_samples, total_samples)
    
    print(f"🎲 Randomly sampling {n_random_samples} from {total_samples} available attack samples...")
    
    # Generate random indices
    random_indices = random.sample(range(total_samples), n_random_samples)
    
    # Sample data and labels
    sampled_samples = attack_samples_cpu[random_indices]
    sampled_labels = attack_labels[random_indices]
    
    print(f"✅ Random sampling completed!")
    print(f"   Original shape: {attack_samples_cpu.shape}")
    print(f"   Sampled shape: {sampled_samples.shape}")
    print(f"   Random indices: {random_indices}")
    
    return sampled_samples, sampled_labels


def manual_reparameterize(mu, logvar):
    """Manual reparameterization trick for VAE (compatible with wrapper models)"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def access_inner_autoencoder_components(model):
    """Helper to access inner autoencoder methods from wrapper model"""
    if hasattr(model, 'structured_autoencoder'):
        return model.structured_autoencoder.decoder, model.structured_autoencoder.apply_affine_transformation
    else:
        # Direct access if not wrapped
        return model.decoder, model.apply_affine_transformation


def generate_non_digit_images(num_images=10, image_size=28, device='cpu'):
    """
    Generate completely non-digit-like images to test what the autoencoder 
    has actually learned. These should look nothing like MNIST digits.
    """
    images = []
    
    # Strategy 1: Geometric patterns
    for i in range(num_images // 5):
        img = torch.zeros(1, image_size, image_size)
        
        # Random geometric shapes
        pattern_type = i % 4
        
        if pattern_type == 0:
            # Concentric circles
            center = image_size // 2
            for radius in range(3, image_size//2, 3):
                y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
                circle_mask = ((x - center) ** 2 + (y - center) ** 2) == radius ** 2
                img[0][circle_mask] = 1.0
                
        elif pattern_type == 1:
            # Diagonal stripes
            for j in range(image_size):
                if j < image_size:
                    img[0, j, j] = 1.0  # Main diagonal
                if j < image_size and (image_size-1-j) >= 0:
                    img[0, j, image_size-1-j] = 1.0  # Anti-diagonal
                    
        elif pattern_type == 2:
            # Checkerboard pattern
            for y in range(0, image_size, 4):
                for x in range(0, image_size, 4):
                    if (y // 4 + x // 4) % 2 == 0:
                        img[0, y:min(y+4, image_size), x:min(x+4, image_size)] = 1.0
                        
        elif pattern_type == 3:
            # Cross pattern
            center = image_size // 2
            img[0, center-2:center+2, :] = 1.0  # Horizontal bar
            img[0, :, center-2:center+2] = 1.0  # Vertical bar
            
        images.append(img)
    
    # Strategy 2: Random noise patterns 
    for i in range(num_images // 5):
        img = torch.zeros(1, image_size, image_size)
        
        # Structured noise
        noise_type = i % 3
        
        if noise_type == 0:
            # Perlin-like noise (approximation)
            x, y = torch.meshgrid(torch.linspace(0, 4*np.pi, image_size), 
                                 torch.linspace(0, 4*np.pi, image_size), indexing='ij')
            img[0] = 0.5 * (torch.sin(x) * torch.cos(y) + 1)
            
        elif noise_type == 1:
            # Random blobs
            for _ in range(5):
                cx, cy = torch.randint(5, image_size-5, (2,))
                size = torch.randint(3, 8, (1,)).item()
                y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
                blob_mask = ((x - cx) ** 2 + (y - cy) ** 2) < size ** 2
                img[0][blob_mask] = torch.rand(1).item()
                
        elif noise_type == 2:
            # Sparse random points
            for _ in range(20):
                px, py = torch.randint(0, image_size, (2,))
                img[0, py, px] = 1.0
                
        images.append(img)
    
    # Strategy 3: Abstract symbols
    for i in range(num_images // 5):
        img = torch.zeros(1, image_size, image_size)
        
        symbol_type = i % 3
        
        if symbol_type == 0:
            # Spiral pattern
            center = image_size // 2
            for angle in np.linspace(0, 4*np.pi, 100):
                radius = angle * 2
                x = int(center + radius * np.cos(angle))
                y = int(center + radius * np.sin(angle))
                if 0 <= x < image_size and 0 <= y < image_size:
                    img[0, y, x] = 1.0
                    
        elif symbol_type == 1:
            # Star pattern
            center = image_size // 2
            for angle in np.linspace(0, 2*np.pi, 8):
                for r in range(5, image_size//2):
                    x = int(center + r * np.cos(angle))
                    y = int(center + r * np.sin(angle))
                    if 0 <= x < image_size and 0 <= y < image_size:
                        img[0, y, x] = 1.0
                        
        elif symbol_type == 2:
            # Grid pattern
            for i in range(0, image_size, 7):
                img[0, i, :] = 1.0  # Horizontal lines
                img[0, :, i] = 1.0  # Vertical lines
                
        images.append(img)
    
    # Strategy 4: Text-like shapes (but not digits)
    for i in range(num_images // 5):
        img = torch.zeros(1, image_size, image_size)
        
        text_type = i % 4
        
        if text_type == 0:
            # H-like shape
            img[0, 5:23, 8:10] = 1.0   # Left vertical
            img[0, 5:23, 18:20] = 1.0  # Right vertical  
            img[0, 13:15, 8:20] = 1.0  # Horizontal bar
            
        elif text_type == 1:
            # X pattern
            for j in range(image_size):
                if 5 <= j <= 22:
                    img[0, j, j-5+5] = 1.0  # Main diagonal
                    img[0, j, 22-j+5] = 1.0  # Anti-diagonal
                    
        elif text_type == 2:
            # Triangle
            for y in range(5, 23):
                width = (y - 5) // 2
                center = image_size // 2
                img[0, y, center-width:center+width+1] = 1.0
                
        elif text_type == 3:
            # Arrow pointing right
            img[0, 13:15, 5:20] = 1.0  # Horizontal line
            for i in range(5):
                img[0, 13-i:15+i, 20-i] = 1.0  # Arrow head
                
        images.append(img)
    
    # Strategy 5: Completely random (baseline)
    for i in range(num_images - len(images)):
        img = torch.zeros(1, image_size, image_size)
        
        # Random sparse pattern
        mask = torch.rand(1, image_size, image_size) > 0.9
        img[mask] = 1.0
        
        images.append(img)
    
    # Convert to tensor and move to device
    all_images = torch.cat(images, dim=0).to(device)
    
    print(f"Generated {len(images)} non-digit images with shape {all_images.shape}")
    print(f"Strategies used: Geometric, Noise, Abstract, Text-like, Random")
    
    return all_images


def analyze_non_digit_reconstructions(model, device, num_samples=10):
    """
    Analyze how the autoencoder reconstructs completely non-digit images.
    This reveals what the model has actually learned - digit-specific features
    vs general image reconstruction capabilities.
    """
    print(f"🔬 Analyzing Non-Digit Reconstructions (n={num_samples})")
    
    # Generate non-digit images
    non_digit_inputs = generate_non_digit_images(num_samples, device=device)
    
    model.eval()
    with torch.no_grad():
        # Get model outputs
        (input_x, content_latent, transform_latent, unused, 
         clean_reconstruction, latent_mu, latent_logvar, final_reconstruction) = model(non_digit_inputs)
    
    # Move to CPU for analysis
    inputs_cpu = non_digit_inputs.cpu()
    clean_recon_cpu = clean_reconstruction.cpu()
    final_recon_cpu = final_reconstruction.cpu()
    content_cpu = content_latent.cpu()
    transform_cpu = transform_latent.cpu()
    
    # Calculate reconstruction metrics
    clean_mses = F.mse_loss(inputs_cpu, clean_recon_cpu, reduction='none').mean(dim=[1,2,3])
    final_mses = F.mse_loss(inputs_cpu, final_recon_cpu, reduction='none').mean(dim=[1,2,3])
    
    # Analyze what "digits" the non-digit images are interpreted as
    # by finding the closest MNIST digit based on reconstruction quality
    
    print(f"\n📊 Reconstruction Quality Analysis:")
    print(f"Clean Reconstruction MSE: {clean_mses.mean():.6f} ± {clean_mses.std():.6f}")
    print(f"Final Reconstruction MSE: {final_mses.mean():.6f} ± {final_mses.std():.6f}")
    
    # Count how many non-digit images have better reconstruction than input MSE vs random
    random_baseline = torch.rand_like(inputs_cpu)
    baseline_mses = F.mse_loss(inputs_cpu, random_baseline, reduction='none').mean(dim=[1,2,3])
    
    better_than_random = (final_mses < baseline_mses).sum().item()
    print(f"\nReconstructions better than random: {better_than_random}/{num_samples} ({100*better_than_random/num_samples:.1f}%)")
    
    # Analyze latent space embeddings
    print(f"\n🧠 Latent Space Analysis:")
    print(f"Content latent stats:")
    print(f"  Mean: [{content_cpu.mean(dim=0)[0]:.3f}, {content_cpu.mean(dim=0)[1]:.3f}]")
    print(f"  Std:  [{content_cpu.std(dim=0)[0]:.3f}, {content_cpu.std(dim=0)[1]:.3f}]")
    print(f"  Range: [{content_cpu.min():.3f}, {content_cpu.max():.3f}]")
    
    print(f"Transform latent stats:")
    print(f"  Mean magnitude: {transform_cpu.norm(dim=1).mean():.3f}")
    print(f"  Std magnitude: {transform_cpu.norm(dim=1).std():.3f}")
    
    # Visualization
    if num_samples <= 12:
        fig, axes = plt.subplots(4, min(num_samples, 6), figsize=(3*min(num_samples, 6), 12))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(min(num_samples, 6)):
            # Original non-digit image
            axes[0, i].imshow(inputs_cpu[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Non-Digit Input {i+1}')
            axes[0, i].axis('off')
            
            # Clean reconstruction (rotated)
            axes[1, i].imshow(np.rot90(clean_recon_cpu[i, 0], k=-1), cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'Clean Recon\nMSE: {clean_mses[i]:.4f}')
            axes[1, i].axis('off')
            
            # Final reconstruction (rotated)
            axes[2, i].imshow(np.rot90(final_recon_cpu[i, 0], k=-1), cmap='gray', vmin=0, vmax=1)
            axes[2, i].set_title(f'Final Recon\nMSE: {final_mses[i]:.4f}')
            axes[2, i].axis('off')
            
            # Latent space info
            axes[3, i].text(0.1, 0.8, f'Content:\n[{content_cpu[i,0]:.2f}, {content_cpu[i,1]:.2f}]', 
                           transform=axes[3, i].transAxes, fontsize=10)
            axes[3, i].text(0.1, 0.4, f'Transform:\n‖θ‖={transform_cpu[i].norm():.2f}', 
                           transform=axes[3, i].transAxes, fontsize=10)
            axes[3, i].set_title('Latent Info')
            axes[3, i].axis('off')
        
        plt.suptitle('🔬 Non-Digit Image Analysis: What Does the Model See?', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.show()
    
    # Summary statistics
    results = {
        'num_samples': num_samples,
        'clean_mse_mean': clean_mses.mean().item(),
        'clean_mse_std': clean_mses.std().item(),
        'final_mse_mean': final_mses.mean().item(),
        'final_mse_std': final_mses.std().item(),
        'better_than_random_count': better_than_random,
        'better_than_random_percent': 100 * better_than_random / num_samples,
        'content_latents': content_cpu.numpy(),
        'transform_latents': transform_cpu.numpy()
    }
    
    print(f"\n✅ Non-digit analysis completed!")
    print(f"Key insight: The model reconstructs non-digit images as {'recognizable' if better_than_random > num_samples//2 else 'poor'} digit-like shapes")
    
    return results, non_digit_inputs, clean_reconstruction, final_reconstruction


def find_optimal_inputs_for_coordinates(model, target_coords, device='cpu', num_iterations=500, lr=0.01):
    """
    Find input images that produce specific content latent coordinates.
    This is inverse engineering - going from desired latent values back to input space.
    """
    print(f"🎯 Finding optimal inputs for target coordinates: {target_coords}")
    
    # Initialize random input image
    input_img = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)
    
    # Use Adam optimizer for better convergence
    optimizer = torch.optim.Adam([input_img], lr=lr)
    
    target_tensor = torch.tensor(target_coords, dtype=torch.float32, device=device)
    
    losses = []
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(torch.clamp(input_img, 0, 1))  # Clamp to valid image range
        content_latent = outputs[1]  # Content latent is second output
        
        # Loss: distance to target coordinates
        loss = F.mse_loss(content_latent.squeeze(), target_tensor)
        
        # Add regularization to encourage realistic images
        reg_loss = 0.01 * torch.mean(torch.abs(input_img))  # L1 regularization
        total_loss = loss + reg_loss
        
        total_loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if iteration % 100 == 0:
            current_coords = content_latent.squeeze().detach().cpu().numpy()
            print(f"Iteration {iteration}: Loss={loss.item():.6f}, Current coords=[{current_coords[0]:.3f}, {current_coords[1]:.3f}]")
    
    # Final result
    final_img = torch.clamp(input_img, 0, 1).detach()
    
    with torch.no_grad():
        final_outputs = model(final_img)
        final_coords = final_outputs[1].squeeze().cpu().numpy()
    
    coord_error = np.linalg.norm(final_coords - target_coords)
    
    print(f"✅ Optimization completed!")
    print(f"Target coordinates: [{target_coords[0]:.3f}, {target_coords[1]:.3f}]")
    print(f"Achieved coordinates: [{final_coords[0]:.3f}, {final_coords[1]:.3f}]")
    print(f"Coordinate error: {coord_error:.6f}")
    
    return final_img, final_coords, losses


def explore_coordinate_space(model, device='cpu'):
    """
    Systematically explore the content latent space by generating inputs
    for different coordinate values and analyzing the results.
    """
    print("🗺️ Exploring Content Latent Coordinate Space")
    
    # Define coordinate grid to explore
    coord_range = np.linspace(-2, 2, 5)  # 5x5 grid from -2 to +2
    
    results = []
    optimized_images = []
    
    print(f"Exploring {len(coord_range)}x{len(coord_range)} = {len(coord_range)**2} coordinate points...")
    
    for i, coord1 in enumerate(coord_range):
        for j, coord2 in enumerate(coord_range):
            target_coords = [coord1, coord2]
            
            print(f"\n--- Exploring coordinate [{coord1:.1f}, {coord2:.1f}] ---")
            
            # Find optimal input for these coordinates
            optimal_img, achieved_coords, losses = find_optimal_inputs_for_coordinates(
                model, target_coords, device, num_iterations=200, lr=0.02
            )
            
            # Analyze the resulting image
            with torch.no_grad():
                outputs = model(optimal_img)
                clean_recon = outputs[4]
                final_recon = outputs[7]
                
                # Calculate reconstruction quality
                recon_mse = F.mse_loss(optimal_img, final_recon).item()
                
                # Analyze image properties
                img_cpu = optimal_img.squeeze().cpu().numpy()
                img_mean = img_cpu.mean()
                img_std = img_cpu.std()
                img_sparsity = (img_cpu < 0.1).mean()  # Fraction of near-zero pixels
            
            result = {
                'target_coords': target_coords,
                'achieved_coords': achieved_coords.tolist(),
                'coord_error': np.linalg.norm(achieved_coords - target_coords),
                'recon_mse': recon_mse,
                'img_mean': img_mean,
                'img_std': img_std,
                'img_sparsity': img_sparsity,
                'final_loss': losses[-1]
            }
            
            results.append(result)
            optimized_images.append(optimal_img.cpu())
    
    # Visualization
    fig, axes = plt.subplots(len(coord_range), len(coord_range), figsize=(12, 12))
    
    for i, coord1 in enumerate(coord_range):
        for j, coord2 in enumerate(coord_range):
            idx = i * len(coord_range) + j
            
            img = optimized_images[idx].squeeze().numpy()
            result = results[idx]
            
            axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i, j].set_title(f'[{coord1:.1f}, {coord2:.1f}]\nErr: {result["coord_error"]:.3f}', fontsize=8)
            axes[i, j].axis('off')
    
    plt.suptitle('🗺️ Content Latent Space Exploration\n(Optimized inputs for each coordinate pair)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()
    
    # Analysis of coordinate space properties
    coord_errors = [r['coord_error'] for r in results]
    recon_mses = [r['recon_mse'] for r in results]
    img_sparsities = [r['img_sparsity'] for r in results]
    
    print(f"\n📊 Coordinate Space Analysis:")
    print(f"Average coordinate error: {np.mean(coord_errors):.6f} ± {np.std(coord_errors):.6f}")
    print(f"Average reconstruction MSE: {np.mean(recon_mses):.6f} ± {np.std(recon_mses):.6f}")
    print(f"Average image sparsity: {np.mean(img_sparsities):.3f} ± {np.std(img_sparsities):.3f}")
    
    # Find best and worst coordinate regions
    best_idx = np.argmin(coord_errors)
    worst_idx = np.argmax(coord_errors)
    
    print(f"\nBest achievable coordinates: {results[best_idx]['target_coords']} (error: {results[best_idx]['coord_error']:.6f})")
    print(f"Worst achievable coordinates: {results[worst_idx]['target_coords']} (error: {results[worst_idx]['coord_error']:.6f})")
    
    if np.mean(coord_errors) < 0.1:
        print("✅ Latent space is highly controllable - can achieve most target coordinates accurately")
    elif np.mean(coord_errors) < 0.5:
        print("⚠️ Latent space is moderately controllable - some coordinates harder to achieve")
    else:
        print("❌ Latent space has limited controllability - many coordinates difficult to achieve")
    
    print(f"\n✅ Coordinate space exploration completed!")
    
    return results, optimized_images


def targeted_content_attack(model, images, epsilon=0.2):
    """
    Perform targeted attack on content latent dimensions specifically.
    This tests how robust the content representation is to adversarial manipulation.
    """
    print(f"🎯 Targeted Content Attack (ε={epsilon})")
    
    images = images.clone().detach().requires_grad_(True)
    
    # Get original content latents
    with torch.no_grad():
        orig_outputs = model(images)
        orig_content = orig_outputs[1]  # Content latent
    
    # Define target content latents (shift towards different regions)
    target_content = orig_content.clone()
    target_content[:, 0] += epsilon * 2  # Shift first content dimension
    target_content[:, 1] -= epsilon * 1.5  # Shift second content dimension
    
    print(f"Original content range: [{orig_content.min():.3f}, {orig_content.max():.3f}]")
    print(f"Target content range: [{target_content.min():.3f}, {target_content.max():.3f}]")
    
    # Iterative attack to achieve target content latents
    adv_images = images.clone()
    
    for iteration in range(10):
        adv_images.requires_grad_(True)
        
        # Forward pass
        outputs = model(adv_images)
        current_content = outputs[1]
        
        # Loss: distance to target content latents
        content_loss = F.mse_loss(current_content, target_content)
        
        # Backward pass
        model.zero_grad()
        content_loss.backward()
        
        # Update adversarial examples
        data_grad = adv_images.grad.data
        adv_images = adv_images + (epsilon/10) * data_grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        
        if iteration % 3 == 0:
            print(f"Iteration {iteration}: Content loss = {content_loss.item():.6f}")
    
    # Final evaluation
    with torch.no_grad():
        final_outputs = model(adv_images)
        final_content = final_outputs[1]
        
        content_shift = F.mse_loss(final_content, orig_content).item()
        target_achievement = F.mse_loss(final_content, target_content).item()
        
        print(f"\n📊 Content Attack Results:")
        print(f"Content shift achieved: {content_shift:.6f}")
        print(f"Distance to target: {target_achievement:.6f}")
        print(f"Image perturbation: {F.mse_loss(adv_images, images).item():.6f}")
    
    return adv_images, final_content, orig_content


def targeted_transform_attack(model, images, epsilon=0.2):
    """
    Perform targeted attack on transform latent dimensions specifically.
    This tests how robust the affine transformation parameters are to adversarial manipulation.
    """
    print(f"🔄 Targeted Transform Attack (ε={epsilon})")
    
    images = images.clone().detach().requires_grad_(True)
    
    # Get original transform latents
    with torch.no_grad():
        orig_outputs = model(images)
        orig_transform = orig_outputs[2]  # Transform latent
    
    # Define target transform latents (introduce specific transformations)
    target_transform = orig_transform.clone()
    # Try to induce rotation by modifying specific transform dimensions
    target_transform[:, 0] += epsilon  # Scale parameter
    target_transform[:, 1] += epsilon * 0.5  # Shear parameter
    target_transform[:, 4] += epsilon * 0.3  # Translation x
    
    print(f"Original transform norms: {orig_transform.norm(dim=1).mean():.3f}")
    print(f"Target transform norms: {target_transform.norm(dim=1).mean():.3f}")
    
    # Iterative attack to achieve target transform latents
    adv_images = images.clone()
    
    for iteration in range(10):
        adv_images.requires_grad_(True)
        
        # Forward pass
        outputs = model(adv_images)
        current_transform = outputs[2]
        
        # Loss: distance to target transform latents
        transform_loss = F.mse_loss(current_transform, target_transform)
        
        # Backward pass
        model.zero_grad()
        transform_loss.backward()
        
        # Update adversarial examples
        data_grad = adv_images.grad.data
        adv_images = adv_images + (epsilon/10) * data_grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        
        if iteration % 3 == 0:
            print(f"Iteration {iteration}: Transform loss = {transform_loss.item():.6f}")
    
    # Final evaluation
    with torch.no_grad():
        final_outputs = model(adv_images)
        final_transform = final_outputs[2]
        
        transform_shift = F.mse_loss(final_transform, orig_transform).item()
        target_achievement = F.mse_loss(final_transform, target_transform).item()
        
        print(f"\n📊 Transform Attack Results:")
        print(f"Transform shift achieved: {transform_shift:.6f}")
        print(f"Distance to target: {target_achievement:.6f}")
        print(f"Image perturbation: {F.mse_loss(adv_images, images).item():.6f}")
    
    return adv_images, final_transform, orig_transform


def evaluate_adversarial_training_benefit(model, samples, labels, epsilon):
    """Simulate the benefit of adversarial training"""
    
    attacker = AffineAdversarialAttacks(model, samples.device)
    
    # Generate adversarial examples
    adv_samples = attacker.fgsm_attack(samples, labels, epsilon)
    
    with torch.no_grad():
        # Clean performance
        _, _, _, _, _, clean_mu, clean_logvar, clean_final = model(samples)
        clean_loss, clean_recon, clean_kl = simplified_affine_kl_loss(
            samples, clean_final, clean_mu, clean_logvar, alpha=1.0, beta=0.008
        )
        
        # Adversarial performance
        _, _, _, _, _, adv_mu, adv_logvar, adv_final = model(adv_samples)
        adv_loss, adv_recon, adv_kl = simplified_affine_kl_loss(
            adv_samples, adv_final, adv_mu, adv_logvar, alpha=1.0, beta=0.008
        )
        
        # Calculate performance degradation
        loss_increase = (adv_loss - clean_loss) / clean_loss * 100
        recon_increase = (adv_recon - clean_recon) / clean_recon * 100
        
    return {
        'clean_loss': clean_loss.item(),
        'adv_loss': adv_loss.item(),
        'loss_increase_percent': loss_increase.item(),
        'recon_increase_percent': recon_increase.item()
    }


def create_synthetic_attack_targets(samples, epsilon=0.1):
    """Create synthetic adversarial examples for transfer testing"""
    
    # Create multiple types of synthetic attacks
    attacks = {}
    
    # Type 1: Random noise attack
    noise = torch.randn_like(samples) * epsilon
    attacks['random_noise'] = torch.clamp(samples + noise, 0, 1)
    
    # Type 2: High-frequency noise (affects edges)
    high_freq_noise = torch.zeros_like(samples)
    for i in range(samples.shape[0]):
        for c in range(samples.shape[1]):
            # Create high-frequency pattern
            freq_pattern = torch.sin(torch.arange(28).float() * 0.5).unsqueeze(0)
            freq_pattern = freq_pattern * freq_pattern.t()
            high_freq_noise[i, c] = freq_pattern * epsilon
    
    attacks['high_freq'] = torch.clamp(samples + high_freq_noise, 0, 1)
    
    # Type 3: Salt and pepper noise
    salt_pepper = torch.rand_like(samples)
    salt_mask = salt_pepper < epsilon/4
    pepper_mask = salt_pepper > (1 - epsilon/4)
    
    salt_pepper_attack = samples.clone()
    salt_pepper_attack[salt_mask] = 1.0
    salt_pepper_attack[pepper_mask] = 0.0
    attacks['salt_pepper'] = salt_pepper_attack
    
    # Type 4: Structured perturbation (grid pattern)
    grid_attack = samples.clone()
    for i in range(0, 28, 4):
        grid_attack[:, :, i, :] = torch.clamp(grid_attack[:, :, i, :] + epsilon, 0, 1)
        grid_attack[:, :, :, i] = torch.clamp(grid_attack[:, :, :, i] + epsilon, 0, 1)
    attacks['grid'] = grid_attack
    
    return attacks


def statistical_attack_analysis(model, test_samples, labels, n_trials=20, epsilon=0.1):
    """
    Perform statistical analysis of attack effectiveness across multiple trials.
    This provides confidence intervals and statistical significance testing.
    """
    print(f"📈 Statistical Attack Analysis (n_trials={n_trials}, ε={epsilon})")
    
    attacker = AffineAdversarialAttacks(model, test_samples.device)
    
    # Storage for results across trials
    fgsm_results = []
    pgd_results = []
    latent_results = []
    
    print("Running multiple attack trials for statistical analysis...")
    
    for trial in range(n_trials):
        if trial % 5 == 0:
            print(f"Trial {trial+1}/{n_trials}")
        
        # Randomly sample from test set for this trial
        batch_size = min(8, len(test_samples))
        indices = torch.randperm(len(test_samples))[:batch_size]
        trial_samples = test_samples[indices]
        trial_labels = labels[indices]
        
        try:
            # FGSM Attack
            fgsm_adv = attacker.fgsm_attack(trial_samples, trial_labels, epsilon)
            with torch.no_grad():
                orig_outputs = model(trial_samples)
                adv_outputs = model(fgsm_adv)
                
                content_shift = F.mse_loss(adv_outputs[1], orig_outputs[1]).item()
                transform_shift = F.mse_loss(adv_outputs[2], orig_outputs[2]).item()
                recon_diff = F.mse_loss(adv_outputs[7], orig_outputs[7]).item()
                
                fgsm_results.append({
                    'content_shift': content_shift,
                    'transform_shift': transform_shift,
                    'recon_diff': recon_diff
                })
            
            # PGD Attack
            pgd_adv = attacker.pgd_attack(trial_samples, trial_labels, epsilon)
            with torch.no_grad():
                adv_outputs = model(pgd_adv)
                
                content_shift = F.mse_loss(adv_outputs[1], orig_outputs[1]).item()
                transform_shift = F.mse_loss(adv_outputs[2], orig_outputs[2]).item()
                recon_diff = F.mse_loss(adv_outputs[7], orig_outputs[7]).item()
                
                pgd_results.append({
                    'content_shift': content_shift,
                    'transform_shift': transform_shift,
                    'recon_diff': recon_diff
                })
            
            # Latent Space Attack
            latent_adv = attacker.latent_space_attack(trial_samples, trial_labels, epsilon)
            with torch.no_grad():
                adv_outputs = model(latent_adv)
                
                content_shift = F.mse_loss(adv_outputs[1], orig_outputs[1]).item()
                transform_shift = F.mse_loss(adv_outputs[2], orig_outputs[2]).item()
                recon_diff = F.mse_loss(adv_outputs[7], orig_outputs[7]).item()
                
                latent_results.append({
                    'content_shift': content_shift,
                    'transform_shift': transform_shift,
                    'recon_diff': recon_diff
                })
                
        except Exception as e:
            print(f"Trial {trial+1} failed: {e}")
            continue
    
    # Statistical analysis
    def analyze_results(results, method_name):
        if not results:
            print(f"No valid results for {method_name}")
            return None
            
        content_shifts = [r['content_shift'] for r in results]
        transform_shifts = [r['transform_shift'] for r in results]
        recon_diffs = [r['recon_diff'] for r in results]
        
        stats = {
            'method': method_name,
            'n_trials': len(results),
            'content_shift_mean': np.mean(content_shifts),
            'content_shift_std': np.std(content_shifts),
            'content_shift_ci': np.percentile(content_shifts, [2.5, 97.5]),
            'transform_shift_mean': np.mean(transform_shifts),
            'transform_shift_std': np.std(transform_shifts),
            'transform_shift_ci': np.percentile(transform_shifts, [2.5, 97.5]),
            'recon_diff_mean': np.mean(recon_diffs),
            'recon_diff_std': np.std(recon_diffs),
            'recon_diff_ci': np.percentile(recon_diffs, [2.5, 97.5])
        }
        
        print(f"\n📊 {method_name} Statistical Results:")
        print(f"  Content Shift: {stats['content_shift_mean']:.6f} ± {stats['content_shift_std']:.6f}")
        print(f"    95% CI: [{stats['content_shift_ci'][0]:.6f}, {stats['content_shift_ci'][1]:.6f}]")
        print(f"  Transform Shift: {stats['transform_shift_mean']:.6f} ± {stats['transform_shift_std']:.6f}")
        print(f"    95% CI: [{stats['transform_shift_ci'][0]:.6f}, {stats['transform_shift_ci'][1]:.6f}]")
        print(f"  Reconstruction Diff: {stats['recon_diff_mean']:.6f} ± {stats['recon_diff_std']:.6f}")
        print(f"    95% CI: [{stats['recon_diff_ci'][0]:.6f}, {stats['recon_diff_ci'][1]:.6f}]")
        
        return stats
    
    # Analyze each attack method
    fgsm_stats = analyze_results(fgsm_results, "FGSM")
    pgd_stats = analyze_results(pgd_results, "PGD")
    latent_stats = analyze_results(latent_results, "Latent Space")
    
    # Comparative analysis
    if fgsm_stats and pgd_stats and latent_stats:
        print(f"\n🔄 Comparative Analysis:")
        
        # Which attack affects content most?
        content_rankings = sorted([
            (fgsm_stats['content_shift_mean'], 'FGSM'),
            (pgd_stats['content_shift_mean'], 'PGD'),
            (latent_stats['content_shift_mean'], 'Latent')
        ], reverse=True)
        
        print(f"Content impact ranking: {' > '.join([f'{name}({val:.4f})' for val, name in content_rankings])}")
        
        # Which attack affects transform most?
        transform_rankings = sorted([
            (fgsm_stats['transform_shift_mean'], 'FGSM'),
            (pgd_stats['transform_shift_mean'], 'PGD'),
            (latent_stats['transform_shift_mean'], 'Latent')
        ], reverse=True)
        
        print(f"Transform impact ranking: {' > '.join([f'{name}({val:.4f})' for val, name in transform_rankings])}")
    
    print(f"\n✅ Statistical analysis completed!")
    
    return {
        'fgsm': fgsm_stats,
        'pgd': pgd_stats,
        'latent': latent_stats,
        'n_trials': n_trials,
        'epsilon': epsilon
    }


def generate_attack_report():
    """
    Generate a comprehensive report of all attack analysis results.
    This function should be called after running all attacks to summarize findings.
    """
    print("📋 Generating Comprehensive Attack Analysis Report")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"adversarial_analysis_report_{timestamp}.txt"
    
    report_content = f"""
ADVERSARIAL ATTACK ANALYSIS REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: Structured Affine-Invariant Autoencoder (2D Content + 6D Transform)

=================================================================
EXECUTIVE SUMMARY
=================================================================

This report summarizes the adversarial robustness analysis of the structured
affine autoencoder, which separates digit content (2D) from affine 
transformations (6D) in its latent representation.

=================================================================
ANALYSIS METHODOLOGY
=================================================================

1. ATTACK METHODS TESTED:
   - FGSM (Fast Gradient Sign Method)
   - PGD (Projected Gradient Descent)  
   - Latent Space Attacks (direct latent manipulation)
   - Targeted Content Attacks
   - Targeted Transform Attacks

2. EVALUATION METRICS:
   - Content Latent Shift (measures semantic impact)
   - Transform Latent Shift (measures geometric impact)
   - Reconstruction Quality Degradation
   - Cross-attack transferability

3. TEST CONDITIONS:
   - Multiple epsilon values: [0.05, 0.1, 0.15, 0.2]
   - Statistical analysis with confidence intervals
   - Non-digit image robustness testing

=================================================================
KEY FINDINGS
=================================================================

[Note: This template should be filled with actual results when called]

CONTENT VS TRANSFORM VULNERABILITY:
- Content latents show [HIGH/MODERATE/LOW] vulnerability to adversarial attacks
- Transform latents demonstrate [HIGHER/SIMILAR/LOWER] robustness compared to content
- Structured separation [DOES/DOES NOT] provide additional robustness

ATTACK METHOD EFFECTIVENESS:
- Most effective attack method: [METHOD NAME]
- Least effective attack method: [METHOD NAME]  
- Attack transferability: [HIGH/MODERATE/LOW]

NON-DIGIT IMAGE ANALYSIS:
- Model interprets non-digit images as: [DIGIT-LIKE/ABSTRACT/NOISE]
- Reconstruction quality on non-digit inputs: [EXCELLENT/GOOD/POOR]
- Latent space behavior: [WELL-STRUCTURED/PARTIALLY-STRUCTURED/CHAOTIC]

=================================================================
RECOMMENDATIONS
=================================================================

1. ADVERSARIAL TRAINING:
   [Recommendation based on vulnerability levels]

2. ARCHITECTURE MODIFICATIONS:
   [Suggestions for improving robustness]

3. DEFENSE STRATEGIES:
   [Specific defense mechanisms to implement]

4. FURTHER RESEARCH:
   [Areas requiring additional investigation]

=================================================================
TECHNICAL DETAILS
=================================================================

[Detailed statistics and technical measurements would be inserted here]

End of Report
"""
    
    # Save report to file
    with open(report_filename, 'w') as f:
        f.write(report_content)
    
    print(f"📄 Report template generated: {report_filename}")
    print("Note: Fill in actual results from your analysis for a complete report")
    
    return report_filename


# Fixed latent space attack function (for compatibility)
def fixed_latent_space_attack(attacker_self, images, labels, epsilon):
    """
    Fixed version of latent space attack that handles the model wrapper correctly.
    This function can be used as a replacement for the original method.
    """
    images = images.clone().detach().requires_grad_(True)
    
    # Get latent representations
    with torch.no_grad():
        (input_x, content_latent, transform_latent, unused, 
         clean_reconstruction, latent_mu, latent_logvar, final_reconstruction) = attacker_self.model(images)
    
    # Create adversarial latent codes
    # Attack content latent (first 2D) more strongly as it affects digit identity
    content_noise = torch.randn_like(latent_mu[:, :2]) * epsilon * 2.0
    transform_noise = torch.randn_like(latent_mu[:, 2:]) * epsilon * 0.5
    
    adv_latent_mu = latent_mu.clone()
    adv_latent_mu[:, :2] += content_noise  # Attack content more
    adv_latent_mu[:, 2:] += transform_noise  # Attack transform less
    
    # Reconstruct from adversarial latent
    with torch.no_grad():
        # Manual reparameterization since wrapper doesn't expose the method
        std = torch.exp(0.5 * latent_logvar)
        eps = torch.randn_like(std)
        adv_latent = adv_latent_mu + eps * std
        
        # Split into content and transform
        adv_content = adv_latent[:, :2]
        adv_transform = adv_latent[:, 2:]
        
        # Decode and apply transform using the inner autoencoder
        decoder, apply_transform = access_inner_autoencoder_components(attacker_self.model)
        adv_clean_recon = decoder(adv_content)
        adv_final_recon = apply_transform(adv_clean_recon, adv_transform)
    
    return adv_final_recon.detach()


if __name__ == "__main__":
    print("Affine Adversarial Attack Analysis Functions Loaded")
    print("Available functions:")
    print("- random_sample_attack_data")
    print("- generate_non_digit_images") 
    print("- analyze_non_digit_reconstructions")
    print("- find_optimal_inputs_for_coordinates")
    print("- explore_coordinate_space")
    print("- targeted_content_attack")
    print("- targeted_transform_attack")
    print("- evaluate_adversarial_training_benefit")
    print("- create_synthetic_attack_targets")
    print("- statistical_attack_analysis")
    print("- generate_attack_report")
    print("- manual_reparameterize")
    print("- access_inner_autoencoder_components")
    print("- fixed_latent_space_attack")
