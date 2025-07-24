"""
Streamlit App for Structured 2D+6D VAE Visualization

This interactive web app allows you to explore pre-trained structured VAE models
with real-time visualization and analysis capabilities.

Features:
- Model loading and configuration
- Interactive latent space grid scanning
- Real-time parameter adjustment
- Random digit reconstruction analysis
- Comprehensive model summaries

Run with: streamlit run streamlit_vae_explorer.py
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from pathlib import Path
import pandas as pd

# Import your modules
try:
    import structured_2d6d_autoencoder as s2d6d
    import affine_autoencoder_shared as shared
    import structured_2d6d_visualization as viz
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Make sure you're running this from the correct directory with all required modules.")
    st.stop()

# Handle torchvision compatibility issues
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchvision::nms.*")

# Fallback data loading function in case of torchvision issues
def get_fallback_mnist_loader(batch_size=128):
    """Fallback MNIST data loader that doesn't rely on torchvision transforms"""
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    except Exception as e:
        st.error(f"Failed to load MNIST data: {e}")
        return None

# Configure Streamlit page
st.set_page_config(
    page_title="Structured VAE Explorer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🎨 Structured VAE Explorer</h1>', unsafe_allow_html=True)
st.markdown("""
**Interactive Visualization for 2D+6D Structured Autoencoders**

Explore pre-trained models with real-time latent space scanning, reconstruction analysis, and comprehensive visualizations.
""")

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'test_loader' not in st.session_state:
    st.session_state.test_loader = None

# Sidebar for configuration and model loading
with st.sidebar:
    st.markdown('<h2 class="section-header">🔧 Configuration</h2>', unsafe_allow_html=True)
    
    # Model loading section
    st.markdown("### 📁 Model Loading")
    
    # Find available .pth files
    current_dir = Path(".")
    pth_files = list(current_dir.glob("*.pth"))
    pth_filenames = [f.name for f in pth_files]
    
    if pth_filenames:
        selected_model = st.selectbox(
            "Select Model File",
            pth_filenames,
            help="Choose a pre-trained model file to load"
        )
        
        if st.button("🚀 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                try:
                    # Setup device
                    device = torch.device('cpu')
                    
                    # Load model
                    checkpoint = torch.load(selected_model, map_location=device, weights_only=False)
                    
                    # Extract config
                    original_config = checkpoint.get('config', {
                        'content_latent_dim': 2,
                        'transform_latent_dim': 6,
                        'total_latent_dim': 8
                    })
                    
                    # Create model
                    model = s2d6d.StructuredAffineInvariantAutoEncoder(
                        content_dim=original_config.get('content_latent_dim', 2),
                        transform_dim=original_config.get('transform_latent_dim', 6)
                    ).to(device)
                    
                    # Load state dict
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    
                    # Setup test data with error handling
                    viz_config = {
                        'batch_size_test': 128,
                        'data_dir': './data',
                        'pin_memory': True,
                        'num_workers': 0  # Set to 0 for Streamlit
                    }
                    
                    test_loader = None
                    try:
                        # Try the original data loading method
                        _, test_loader = shared.get_cloud_mnist_loaders(**viz_config)
                    except Exception as data_error:
                        st.warning(f"Standard data loading failed: {data_error}")
                        try:
                            # Fallback to simple MNIST loading
                            test_loader = get_fallback_mnist_loader(viz_config['batch_size_test'])
                        except Exception as fallback_error:
                            st.error(f"Fallback data loading also failed: {fallback_error}")
                            st.info("Please ensure MNIST data is available or check your torchvision installation.")
                    
                    if test_loader is None:
                        st.error("Could not load test data. Please check your data directory and dependencies.")
                        st.session_state.model_loaded = False
                        st.stop()
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.test_loader = test_loader
                    st.session_state.device = device
                    st.session_state.original_config = original_config
                    st.session_state.losses_dict = checkpoint.get('losses', {})
                    st.session_state.model_loaded = True
                    
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Failed to load model: {e}")
                    st.session_state.model_loaded = False
    else:
        st.warning("No .pth files found in current directory")
        st.info("Please ensure your model files are in the same directory as this script")
    
    # Visualization parameters
    if st.session_state.model_loaded:
        st.markdown("### 🎛️ Visualization Parameters")
        
        grid_size = st.slider(
            "Grid Size",
            min_value=10,
            max_value=50,
            value=20,
            help="Size of the latent space grid (NxN)"
        )
        
        latent_range = st.slider(
            "Latent Range",
            min_value=1.0,
            max_value=8.0,
            value=3.0,
            step=0.5,
            help="Range for latent space scanning (-range to +range)"
        )
        
        num_samples = st.slider(
            "Number of Random Samples",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of random digits to analyze"
        )
        
        rotate_images = st.checkbox(
            "Rotate Images 90°",
            value=True,
            help="Apply 90-degree clockwise rotation to images"
        )

# Main content area
if not st.session_state.model_loaded:
    st.info("👈 Please load a model from the sidebar to begin visualization")
    
    # Show some information about the app
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Features
        - **Interactive Grid Scanning**: Explore the 2D content latent space
        - **Real-time Visualization**: Adjust parameters and see results instantly
        - **Reconstruction Analysis**: Compare original vs reconstructed digits
        """)
    
    with col2:
        st.markdown("""
        ### 🏗️ Model Architecture
        - **2D Content Latent**: Captures digit identity/shape
        - **6D Transform Latent**: Handles spatial transformations
        - **Affine Invariance**: Separates content from geometric transforms
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Analysis Tools
        - **Latent Space Diagnostics**: Understand digit clustering
        - **Quality Metrics**: Reconstruction error analysis
        - **Interactive Controls**: Customize all visualizations
        """)

else:
    # Model information section
    st.markdown('<h2 class="section-header">📊 Model Information</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_params = sum(p.numel() for p in st.session_state.model.parameters())
        st.metric("Total Parameters", f"{total_params:,}")
    
    with col2:
        content_dim = st.session_state.original_config.get('content_latent_dim', 2)
        st.metric("Content Latent Dim", content_dim)
    
    with col3:
        transform_dim = st.session_state.original_config.get('transform_latent_dim', 6)
        st.metric("Transform Latent Dim", transform_dim)
    
    with col4:
        alpha = st.session_state.original_config.get('alpha', '?')
        st.metric("Alpha (Affine Weight)", alpha)
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Latent Space Grid", "🎲 Random Reconstructions", "🔍 Latent Analysis", "📈 Model Details"])
    
    with tab1:
        st.markdown("### Latent Space Grid Scan")
        st.markdown("Explore how different regions of the 2D content latent space generate different digit types.")
        
        if st.button("🎨 Generate Grid Scan", type="primary"):
            with st.spinner("Generating latent space grid..."):
                try:
                    # Generate grid
                    grid_images, x_vals, y_vals = viz.create_latent_space_grid(
                        st.session_state.model,
                        st.session_state.device,
                        grid_size=grid_size,
                        latent_range=latent_range,
                        rotate_images=rotate_images
                    )
                    
                    # Create matplotlib figure
                    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
                    fig.suptitle('🗺️ 2D Content Latent Space Grid Scan', fontsize=16, y=0.95)
                    
                    for i in range(grid_size):
                        for j in range(grid_size):
                            axes[i, j].imshow(grid_images[i, j], cmap='gray')
                            axes[i, j].set_xticks([])
                            axes[i, j].set_yticks([])
                            
                            # Add coordinates for edge cases
                            if i == 0:
                                axes[i, j].set_title(f'{x_vals[j]:.1f}', fontsize=8, pad=2)
                            if j == 0:
                                axes[i, j].set_ylabel(f'{y_vals[i]:.1f}', fontsize=8, rotation=0, ha='right')
                    
                    # Add axis labels
                    fig.text(0.5, 0.02, 'Content Latent Dimension 1 →', ha='center', fontsize=12)
                    fig.text(0.02, 0.5, 'Content Latent Dimension 2 →', va='center', rotation=90, fontsize=12)
                    
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.95)
                    
                    # Display in Streamlit
                    st.pyplot(fig)
                    
                    st.success(f"✅ Generated {grid_size}×{grid_size} = {grid_size**2} images")
                    st.info(f"📏 Scanned range: [{-latent_range:.1f}, {latent_range:.1f}] in both dimensions")
                    
                except Exception as e:
                    st.error(f"Error generating grid: {e}")
    
    with tab2:
        st.markdown("### Random Digit Reconstruction Analysis")
        st.markdown("Sample random digits and compare original, latent-decoded, and complete reconstructions.")
        
        if st.button("🎲 Analyze Random Reconstructions", type="primary"):
            with st.spinner("Analyzing random reconstructions..."):
                try:
                    # Collect random samples
                    model = st.session_state.model
                    test_loader = st.session_state.test_loader
                    device = st.session_state.device
                    
                    model.eval()
                    sampled_data = []
                    sampled_labels = []
                    
                    with torch.no_grad():
                        for data, labels in test_loader:
                            data = data.to(device)
                            batch_size = data.shape[0]
                            
                            if len(sampled_data) < num_samples:
                                needed = min(num_samples - len(sampled_data), batch_size)
                                random_indices = torch.randperm(batch_size)[:needed]
                                sampled_data.append(data[random_indices])
                                sampled_labels.extend(labels[random_indices].tolist())
                            
                            if len(sampled_data) >= num_samples:
                                break
                        
                        # Process samples
                        sample_batch = torch.cat(sampled_data)[:num_samples]
                        sample_labels = sampled_labels[:num_samples]
                        
                        # Get model outputs
                        model_output = model(sample_batch)
                        complete_reconstruction = model_output[0]
                        content_latents = model_output[1]
                        latent_decoded = model.structured_autoencoder.decoder(content_latents)
                        
                        # Create visualization
                        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples*1.5, 6))
                        
                        for i in range(num_samples):
                            # Apply rotation if requested
                            if rotate_images:
                                original_img = np.rot90(sample_batch[i].cpu().squeeze().numpy(), k=-1)
                                latent_img = np.rot90(latent_decoded[i].cpu().squeeze().numpy(), k=-1)
                                complete_img = np.rot90(complete_reconstruction[i].cpu().squeeze().numpy(), k=-1)
                            else:
                                original_img = sample_batch[i].cpu().squeeze().numpy()
                                latent_img = latent_decoded[i].cpu().squeeze().numpy()
                                complete_img = complete_reconstruction[i].cpu().squeeze().numpy()
                            
                            # Plot images
                            axes[0, i].imshow(original_img, cmap='gray')
                            axes[0, i].set_title(f'Digit {sample_labels[i]}', fontsize=10)
                            axes[0, i].set_xticks([])
                            axes[0, i].set_yticks([])
                            
                            axes[1, i].imshow(latent_img, cmap='gray')
                            axes[1, i].set_xticks([])
                            axes[1, i].set_yticks([])
                            
                            axes[2, i].imshow(complete_img, cmap='gray')
                            axes[2, i].set_xticks([])
                            axes[2, i].set_yticks([])
                            
                            # Show coordinates
                            content_coord = content_latents[i].cpu().numpy()
                            axes[1, i].set_xlabel(f'({content_coord[0]:.1f}, {content_coord[1]:.1f})', fontsize=8)
                        
                        # Add row labels
                        axes[0, 0].set_ylabel('Original\nMNIST', rotation=90, fontsize=12, ha='right')
                        axes[1, 0].set_ylabel('Latent\nDecoded', rotation=90, fontsize=12, ha='right')
                        axes[2, 0].set_ylabel('Complete\nReconstruction', rotation=90, fontsize=12, ha='right')
                        
                        plt.suptitle('🎲 Random Digit Reconstruction Pipeline', fontsize=16, y=0.95)
                        plt.tight_layout()
                        plt.subplots_adjust(top=0.88, left=0.12)
                        
                        st.pyplot(fig)
                        
                        # Calculate and display metrics
                        latent_mse = torch.mean((sample_batch - latent_decoded)**2, dim=[1,2,3])
                        complete_mse = torch.mean((sample_batch - complete_reconstruction)**2, dim=[1,2,3])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Latent Reconstruction MSE", f"{latent_mse.mean():.4f}")
                        with col2:
                            st.metric("Complete Reconstruction MSE", f"{complete_mse.mean():.4f}")
                        with col3:
                            improvement = ((latent_mse.mean() - complete_mse.mean()) / latent_mse.mean() * 100)
                            st.metric("Improvement from Affine", f"{improvement:.1f}%")
                        
                        # Show per-digit analysis
                        st.markdown("#### Per-Digit Analysis")
                        analysis_data = []
                        for i, label in enumerate(sample_labels):
                            content_coord = content_latents[i].cpu().numpy()
                            analysis_data.append({
                                'Digit': label,
                                'Content X': f"{content_coord[0]:.2f}",
                                'Content Y': f"{content_coord[1]:.2f}",
                                'Latent MSE': f"{latent_mse[i]:.4f}",
                                'Complete MSE': f"{complete_mse[i]:.4f}"
                            })
                        
                        df = pd.DataFrame(analysis_data)
                        st.dataframe(df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error in reconstruction analysis: {e}")
    
    with tab3:
        st.markdown("### Latent Space Analysis")
        st.markdown("Analyze the distribution of real MNIST digits in the latent space.")
        
        if st.button("🔍 Run Latent Analysis", type="primary"):
            with st.spinner("Analyzing latent space distribution..."):
                try:
                    content_data, labels, suggested_range = viz.diagnose_latent_range(
                        st.session_state.model,
                        st.session_state.test_loader,
                        st.session_state.device,
                        max_batches=10
                    )
                    
                    # Create scatter plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(content_data[:, 0], content_data[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
                    plt.colorbar(scatter, label='Digit')
                    ax.set_title('🗺️ Actual MNIST Digit Locations in 2D Content Latent Space')
                    ax.set_xlabel('Content Latent Dimension 1')
                    ax.set_ylabel('Content Latent Dimension 2')
                    ax.grid(True, alpha=0.3)
                    
                    # Add digit labels
                    for digit in range(10):
                        digit_mask = labels == digit
                        if np.sum(digit_mask) > 0:
                            digit_latents = content_data[digit_mask]
                            center_x = np.mean(digit_latents[:, 0])
                            center_y = np.mean(digit_latents[:, 1])
                            ax.annotate(str(digit), (center_x, center_y), fontsize=12, fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display range information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current Grid Range", latent_range)
                    with col2:
                        st.metric("Suggested Grid Range", f"{suggested_range:.1f}")
                    
                    if suggested_range > latent_range:
                        st.info(f"💡 Consider increasing grid range to {suggested_range:.1f} to capture all digit locations")
                    
                except Exception as e:
                    st.error(f"Error in latent analysis: {e}")
    
    with tab4:
        st.markdown("### Model Configuration and Training Details")
        
        # Configuration details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Configuration")
            config_data = {
                'Parameter': list(st.session_state.original_config.keys()),
                'Value': list(st.session_state.original_config.values())
            }
            st.dataframe(pd.DataFrame(config_data), use_container_width=True)
        
        with col2:
            st.markdown("#### Training Information")
            losses_dict = st.session_state.losses_dict
            if losses_dict:
                if 'total_losses' in losses_dict and losses_dict['total_losses']:
                    st.metric("Final Training Loss", f"{losses_dict['total_losses'][-1]:.4f}")
                    st.metric("Epochs Trained", len(losses_dict.get('total_losses', [])))
                    
                    # Plot training curves if available
                    if len(losses_dict.get('total_losses', [])) > 1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if 'total_losses' in losses_dict:
                            ax.plot(losses_dict['total_losses'], label='Total Loss', linewidth=2)
                        if 'recon_losses' in losses_dict:
                            ax.plot(losses_dict['recon_losses'], label='Reconstruction Loss', linewidth=2)
                        if 'kl_losses' in losses_dict:
                            ax.plot(losses_dict['kl_losses'], label='KL Loss', linewidth=2)
                        
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title('Training Progress')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                else:
                    st.info("No training loss data available")
            else:
                st.info("No training information available")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎨 Structured VAE Explorer | Built with Streamlit | Interactive ML Visualization</p>
</div>
""", unsafe_allow_html=True)
