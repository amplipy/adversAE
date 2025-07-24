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
