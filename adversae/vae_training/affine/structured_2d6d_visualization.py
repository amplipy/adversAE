"""
Structured 2D+6D Model Visualization Functions

This module contains all the visualization functions for analyzing pre-trained
2D+6D structured autoencoders. Functions are designed to be imported and used
in visualization notebooks or scripts.

Functions:
- create_latent_space_grid: Generate grid scan of 2D content latent space
- diagnose_latent_range: Analyze actual latent space distribution
- analyze_random_reconstructions: Sample and analyze digit reconstructions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


def create_latent_space_grid(model, device, grid_size=10, latent_range=3.0, transform_latent=None, rotate_images=True):
    """
    Create a grid of generated digits by scanning the 2D content latent space
    
    Args:
        model: Trained StructuredAffineInvariantAutoEncoder
        device: torch device
        grid_size: Size of the grid (grid_size x grid_size)
        latent_range: Range of latent values to scan (-range to +range)
        transform_latent: Fixed 6D transform latent (None for zeros)
        rotate_images: Whether to rotate images 90 degrees clockwise
    
    Returns:
        tuple: (grid_images, x_vals, y_vals)
    """
    model.eval()
    
    # Create grid of 2D content latent values
    x_vals = np.linspace(-latent_range, latent_range, grid_size)
    y_vals = np.linspace(-latent_range, latent_range, grid_size)
    
    # Generate images for each grid point
    grid_images = []
    
    with torch.no_grad():
        for i, y in enumerate(y_vals):
            row_images = []
            for j, x in enumerate(x_vals):
                # Create 2D content latent
                content_latent = torch.tensor([[x, y]], dtype=torch.float32, device=device)
                
                # Create 6D transform latent (zeros or custom)
                if transform_latent is None:
                    transform_latent_tensor = torch.zeros(1, 6, device=device)
                else:
                    transform_latent_tensor = torch.tensor([transform_latent], dtype=torch.float32, device=device)
                
                # Use only the 2D content latent for decoding
                # The decoder expects only 2D content latent, not combined 8D
                clean_digit = model.structured_autoencoder.decoder(content_latent)
                
                # Apply the affine transformation separately
                # For now, let's use the clean digit directly (transform applied to zeros)
                final_image = clean_digit
                
                # Convert to numpy for plotting
                img = final_image.cpu().squeeze().numpy()
                
                # Optionally rotate image 90 degrees to the right (clockwise)
                if rotate_images:
                    img = np.rot90(img, k=-1)  # k=-1 for clockwise rotation
                
                row_images.append(img)
            
            grid_images.append(row_images)
    
    return np.array(grid_images), x_vals, y_vals


def plot_latent_space_grid(grid_images, x_vals, y_vals, grid_size, latent_range, title="🗺️ 2D Content Latent Space Grid Scan"):
    """
    Plot the latent space grid scan results
    
    Args:
        grid_images: Array of generated images
        x_vals: X-axis values for the grid
        y_vals: Y-axis values for the grid
        grid_size: Size of the grid
        latent_range: Range of latent values scanned
        title: Title for the plot
    """
    # Plot the grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(title, fontsize=16, y=0.95)
    
    for i in range(grid_size):
        for j in range(grid_size):
            axes[i, j].imshow(grid_images[i, j], cmap='gray')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            
            # Add latent coordinates as titles for edge cases
            if i == 0:  # Top row
                axes[i, j].set_title(f'{x_vals[j]:.1f}', fontsize=8, pad=2)
            if j == 0:  # Left column  
                axes[i, j].set_ylabel(f'{y_vals[i]:.1f}', fontsize=8, rotation=0, ha='right')
    
    # Add axis labels
    fig.text(0.5, 0.02, 'Content Latent Dimension 1 →', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Content Latent Dimension 2 →', va='center', rotation=90, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.95)
    plt.show()
    
    print(f"✅ Grid scan complete! Generated {grid_size}×{grid_size} = {grid_size**2} images")
    print(f"📏 Scanned range: [{-latent_range:.1f}, {latent_range:.1f}] in both dimensions")


def diagnose_latent_range(model, test_loader, device, max_batches=10):
    """
    Find the actual range where real MNIST digits exist in latent space
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: torch device
        max_batches: Maximum number of batches to process
    
    Returns:
        tuple: (content_latents, labels, suggested_range)
    """
    model.eval()
    all_content_latents = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            if batch_idx >= max_batches:  # Limit for speed
                break
                
            data = data.to(device)
            
            # Get model output - we now know it returns 7 items
            model_output = model(data)
            
            # Extract the 2D content latent (Item 1)
            z_content = model_output[1]  # Shape: [batch_size, 2]
            
            all_content_latents.append(z_content.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Combine all batches
    content_latents = np.vstack(all_content_latents)
    labels = np.hstack(all_labels)
    
    # Analyze range for each digit
    print("🔍 ACTUAL LATENT SPACE RANGES FOR EACH DIGIT:")
    print("=" * 60)
    
    for digit in range(10):
        digit_mask = labels == digit
        if np.sum(digit_mask) > 0:
            digit_latents = content_latents[digit_mask]
            dim1_range = [np.min(digit_latents[:, 0]), np.max(digit_latents[:, 0])]
            dim2_range = [np.min(digit_latents[:, 1]), np.max(digit_latents[:, 1])]
            print(f"Digit {digit}: Dim1 [{dim1_range[0]:.2f}, {dim1_range[1]:.2f}], Dim2 [{dim2_range[0]:.2f}, {dim2_range[1]:.2f}]")
    
    # Overall statistics
    overall_min = [np.min(content_latents[:, 0]), np.min(content_latents[:, 1])]
    overall_max = [np.max(content_latents[:, 0]), np.max(content_latents[:, 1])]
    overall_mean = [np.mean(content_latents[:, 0]), np.mean(content_latents[:, 1])]
    overall_std = [np.std(content_latents[:, 0]), np.std(content_latents[:, 1])]
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"Full Range: Dim1 [{overall_min[0]:.2f}, {overall_max[0]:.2f}], Dim2 [{overall_min[1]:.2f}, {overall_max[1]:.2f}]")
    print(f"Mean: Dim1 {overall_mean[0]:.2f}, Dim2 {overall_mean[1]:.2f}")
    print(f"Std:  Dim1 {overall_std[0]:.2f}, Dim2 {overall_std[1]:.2f}")
    
    # Recommend grid scan range
    suggested_range = max(abs(overall_min[0]), abs(overall_max[0]), abs(overall_min[1]), abs(overall_max[1])) * 1.1
    print(f"\n💡 SUGGESTED GRID SCAN RANGE: {suggested_range:.1f}")
    print(f"   (This would scan [{-suggested_range:.1f}, {suggested_range:.1f}] in both dimensions)")
    
    return content_latents, labels, suggested_range


def plot_latent_scatter(content_data, labels, title="🗺️ Actual MNIST Digit Locations in 2D Content Latent Space"):
    """
    Create a scatter plot to visualize where digits actually are in latent space
    
    Args:
        content_data: Array of content latent coordinates
        labels: Array of digit labels
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(content_data[:, 0], content_data[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Digit')
    plt.title(title)
    plt.xlabel('Content Latent Dimension 1')
    plt.ylabel('Content Latent Dimension 2')
    plt.grid(True, alpha=0.3)
    
    # Add digit labels for clarity
    for digit in range(10):
        digit_mask = labels == digit
        if np.sum(digit_mask) > 0:
            digit_latents = content_data[digit_mask]
            center_x = np.mean(digit_latents[:, 0])
            center_y = np.mean(digit_latents[:, 1])
            plt.annotate(str(digit), (center_x, center_y), fontsize=12, fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def analyze_random_reconstructions(model, test_loader, device, num_samples=10, rotate_images=True):
    """
    Sample random digits and show their reconstruction pipeline:
    Top row: Original digits
    Middle row: Latent space decoded (content only)
    Bottom row: Complete reconstruction (with affine transform)
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: torch device
        num_samples: Number of random samples to analyze
        rotate_images: Whether to rotate images for consistency
    """
    model.eval()
    
    # Collect random samples
    sampled_data = []
    sampled_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            
            # Randomly select indices from this batch
            batch_size = data.shape[0]
            if len(sampled_data) < num_samples:
                needed = min(num_samples - len(sampled_data), batch_size)
                random_indices = torch.randperm(batch_size)[:needed]
                
                sampled_data.append(data[random_indices])
                sampled_labels.extend(labels[random_indices].tolist())
            
            if len(sampled_data) >= num_samples:
                break
        
        # Combine all samples
        sample_batch = torch.cat(sampled_data)[:num_samples]
        sample_labels = sampled_labels[:num_samples]
        
        # Get model outputs for the samples
        model_output = model(sample_batch)
        
        # Extract components
        complete_reconstruction = model_output[0]  # Final output after affine transform
        content_latents = model_output[1]          # 2D content latent
        
        # Generate latent-only reconstructions (content decoder without affine)
        latent_decoded = model.structured_autoencoder.decoder(content_latents)
        
        # Create visualization
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples*1.5, 6))
        
        for i in range(num_samples):
            # Apply rotation to all images for consistency with grid scan
            if rotate_images:
                original_img = np.rot90(sample_batch[i].cpu().squeeze().numpy(), k=-1)
                latent_img = np.rot90(latent_decoded[i].cpu().squeeze().numpy(), k=-1)
                complete_img = np.rot90(complete_reconstruction[i].cpu().squeeze().numpy(), k=-1)
            else:
                original_img = sample_batch[i].cpu().squeeze().numpy()
                latent_img = latent_decoded[i].cpu().squeeze().numpy()
                complete_img = complete_reconstruction[i].cpu().squeeze().numpy()
            
            # Top row: Original digits
            axes[0, i].imshow(original_img, cmap='gray')
            axes[0, i].set_title(f'Digit {sample_labels[i]}', fontsize=10)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            # Middle row: Latent decoded (content only)
            axes[1, i].imshow(latent_img, cmap='gray')
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            
            # Bottom row: Complete reconstruction
            axes[2, i].imshow(complete_img, cmap='gray')
            axes[2, i].set_xticks([])
            axes[2, i].set_yticks([])
            
            # Show latent coordinates
            content_coord = content_latents[i].cpu().numpy()
            axes[1, i].set_xlabel(f'({content_coord[0]:.1f}, {content_coord[1]:.1f})', fontsize=8)
        
        # Add row labels
        axes[0, 0].set_ylabel('Original\nMNIST', rotation=90, fontsize=12, ha='right')
        axes[1, 0].set_ylabel('Latent\nDecoded', rotation=90, fontsize=12, ha='right')
        axes[2, 0].set_ylabel('Complete\nReconstruction', rotation=90, fontsize=12, ha='right')
        
        plt.suptitle('🎲 Random Digit Reconstruction Pipeline', fontsize=16, y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, left=0.12)
        plt.show()
        
        # Print reconstruction quality metrics
        print(f"\n📊 RECONSTRUCTION QUALITY ANALYSIS:")
        print("="*60)
        
        # Calculate MSE for latent vs complete reconstructions
        latent_mse = torch.mean((sample_batch - latent_decoded)**2, dim=[1,2,3])
        complete_mse = torch.mean((sample_batch - complete_reconstruction)**2, dim=[1,2,3])
        
        print(f"Average Latent Reconstruction MSE: {latent_mse.mean():.4f}")
        print(f"Average Complete Reconstruction MSE: {complete_mse.mean():.4f}")
        print(f"Improvement from Affine Transform: {((latent_mse.mean() - complete_mse.mean()) / latent_mse.mean() * 100):.1f}%")
        
        # Show individual digit analysis
        print(f"\n🔢 PER-DIGIT ANALYSIS:")
        for i, label in enumerate(sample_labels):
            content_coord = content_latents[i].cpu().numpy()
            print(f"Digit {label}: Content=({content_coord[0]:.2f}, {content_coord[1]:.2f}), "
                  f"Latent MSE={latent_mse[i]:.4f}, Complete MSE={complete_mse[i]:.4f}")


def run_complete_latent_analysis(model, test_loader, device, viz_config):
    """
    Run the complete latent space analysis including diagnostic and grid scan
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: torch device
        viz_config: Configuration dictionary with grid_size, latent_range, etc.
    """
    print("🔍 Running complete latent space analysis...")
    
    # Step 1: Diagnose actual latent range
    print("\n🔍 Diagnosing actual latent space distribution...")
    content_data, labels, suggested_range = diagnose_latent_range(model, test_loader, device)
    
    # Step 2: Create scatter plot
    plot_latent_scatter(content_data, labels)
    
    # Step 3: Update config with suggested range if desired
    print(f"\n🔧 UPDATING GRID SCAN RANGE:")
    print(f"Current range: {viz_config['latent_range']}")
    print(f"Suggested range: {suggested_range:.1f}")
    
    # Step 4: Create and plot grid scan
    print("\n🎨 Creating latent space grid scan...")
    grid_images, x_vals, y_vals = create_latent_space_grid(
        model, device, 
        grid_size=viz_config['grid_size'], 
        latent_range=viz_config['latent_range']
    )
    
    plot_latent_space_grid(grid_images, x_vals, y_vals, 
                          viz_config['grid_size'], viz_config['latent_range'])
    
    return content_data, labels, suggested_range


def print_model_summary(model, losses_dict, original_config):
    """
    Print a comprehensive model summary
    
    Args:
        model: Trained model
        losses_dict: Dictionary of training losses
        original_config: Model configuration dictionary
    """
    print("🔍 MODEL SUMMARY")
    print("="*50)
    
    # Model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🎛️ Trainable parameters: {trainable_params:,}")
    
    # Training info if available
    if losses_dict:
        if 'total_losses' in losses_dict and losses_dict['total_losses']:
            final_loss = losses_dict['total_losses'][-1]
            print(f"📉 Final training loss: {final_loss:.4f}")
        print(f"🏃 Epochs trained: {len(losses_dict.get('total_losses', []))}")
    
    # Configuration
    print(f"⚙️ Alpha (affine weight): {original_config.get('alpha', '?')}")
    print(f"⚙️ Beta (KL weight): {original_config.get('beta', '?')}")
    print(f"🎯 Simplified loss function: ✅")
    
    print("\n✅ Ready for visualization!")
