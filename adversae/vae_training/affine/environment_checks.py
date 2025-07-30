"""
Environment Check Functions for Adversarial Autoencoder Analysis

This module aggregates all environmental check functions used in the adversarial 
attack analysis notebook. It provides comprehensive system compatibility checks,
device configuration, and fallback mechanisms.

Created: July 27, 2025
"""

import os
import sys
import torch
import platform
from typing import Dict, Any, Tuple, Optional
import warnings

# Import project modules with fallback
try:
    import affine_autoencoder_shared as shared
except ImportError:
    shared = None
    warnings.warn("affine_autoencoder_shared not available, using fallback methods")


class EnvironmentChecker:
    """Comprehensive environment checking and setup utility."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the environment checker.
        
        Args:
            config: Configuration dictionary with device preferences and settings
        """
        self.config = config or self._get_default_config()
        self.results = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if none provided."""
        return {
            'device_preference': 'cuda',
            'data_dir': '../data',
            'batch_size_test': 64,
            'enable_mps_fallback': True,
            'verbose': True
        }
    
    def run_full_environment_check(self) -> Dict[str, Any]:
        """
        Run comprehensive environment check covering all aspects.
        
        Returns:
            Dictionary with check results and recommendations
        """
        if self.config.get('verbose', True):
            print("🔍 Running comprehensive environment check...")
            
        results = {
            'system_info': self.check_system_info(),
            'python_info': self.check_python_environment(),
            'pytorch_info': self.check_pytorch_environment(),
            'device_info': self.check_device_availability(),
            'compatibility': self.check_device_compatibility(),
            'data_setup': self.check_data_directory(),
            'recommendations': []
        }
        
        # Generate recommendations based on results
        results['recommendations'] = self._generate_recommendations(results)
        
        if self.config.get('verbose', True):
            self._print_summary(results)
            
        self.results = results
        return results
    
    def check_system_info(self) -> Dict[str, Any]:
        """Check basic system information."""
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'architecture': platform.architecture()[0]
        }
    
    def check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment and key package versions."""
        info = {
            'python_version': sys.version.split()[0],
            'executable': sys.executable,
            'path': sys.path[:3],  # First 3 entries
            'packages': {}
        }
        
        # Check key packages
        packages_to_check = ['torch', 'torchvision', 'numpy', 'matplotlib', 'tqdm']
        
        for package in packages_to_check:
            try:
                module = __import__(package)
                info['packages'][package] = getattr(module, '__version__', 'Unknown')
            except ImportError:
                info['packages'][package] = 'Not installed'
                
        return info
    
    def check_pytorch_environment(self) -> Dict[str, Any]:
        """Check PyTorch-specific environment details."""
        try:
            import torch
            
            info = {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
                'num_threads': torch.get_num_threads(),
                'backends': {
                    'cudnn': torch.backends.cudnn.enabled if torch.cuda.is_available() else False,
                    'mkldnn': torch.backends.mkldnn.enabled,
                }
            }
            
            # Get CUDA device names if available
            if info['cuda_available']:
                info['cuda_devices'] = []
                for i in range(info['cuda_device_count']):
                    info['cuda_devices'].append({
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_total': torch.cuda.get_device_properties(i).total_memory,
                        'memory_cached': torch.cuda.memory_cached(i) if torch.cuda.is_available() else 0
                    })
                    
        except ImportError:
            info = {'error': 'PyTorch not available'}
            
        return info
    
    def check_device_availability(self) -> Dict[str, Any]:
        """Check available computing devices and their capabilities."""
        devices = {
            'cpu': {'available': True, 'primary': False},
            'cuda': {'available': False, 'primary': False},
            'mps': {'available': False, 'primary': False}
        }
        
        try:
            import torch
            
            # Check CUDA
            if torch.cuda.is_available():
                devices['cuda']['available'] = True
                devices['cuda']['device_count'] = torch.cuda.device_count()
                devices['cuda']['current_device'] = torch.cuda.current_device()
                
            # Check MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                devices['mps']['available'] = True
                
            # Determine primary device based on preference
            preference = self.config.get('device_preference', 'cuda')
            if preference == 'cuda' and devices['cuda']['available']:
                devices['cuda']['primary'] = True
            elif preference == 'mps' and devices['mps']['available']:
                devices['mps']['primary'] = True
            else:
                devices['cpu']['primary'] = True
                
        except ImportError:
            pass
            
        return devices
    
    def check_device_compatibility(self) -> Dict[str, Any]:
        """Check device compatibility for specific operations (like adversarial attacks)."""
        compatibility = {
            'mps_grid_sampler': False,
            'cuda_operations': False,
            'recommended_device': 'cpu',
            'issues': []
        }
        
        try:
            import torch
            
            # Test CUDA compatibility
            if torch.cuda.is_available():
                try:
                    # Basic CUDA operation test
                    test_tensor = torch.randn(2, 2).cuda()
                    _ = test_tensor @ test_tensor.T
                    compatibility['cuda_operations'] = True
                except Exception as e:
                    compatibility['issues'].append(f"CUDA operation failed: {e}")
            
            # Test MPS compatibility for grid sampling (important for affine transformations)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    # Test grid_sample operation which is used in affine transformations
                    device = torch.device('mps')
                    test_input = torch.randn(1, 1, 4, 4, device=device)
                    grid = torch.randn(1, 2, 2, 2, device=device)
                    _ = torch.nn.functional.grid_sample(test_input, grid, align_corners=False)
                    compatibility['mps_grid_sampler'] = True
                except Exception as e:
                    compatibility['issues'].append(f"MPS grid_sample failed: {e}")
            
            # Determine recommended device
            if compatibility['cuda_operations']:
                compatibility['recommended_device'] = 'cuda'
            elif compatibility['mps_grid_sampler']:
                compatibility['recommended_device'] = 'mps'
            else:
                compatibility['recommended_device'] = 'cpu'
                
        except ImportError:
            compatibility['issues'].append("PyTorch not available")
            
        return compatibility
    
    def check_data_directory(self) -> Dict[str, Any]:
        """Check data directory setup and accessibility."""
        data_dir = self.config.get('data_dir', '../data')
        
        info = {
            'path': data_dir,
            'exists': os.path.exists(data_dir),
            'readable': False,
            'writable': False,
            'size_mb': 0
        }
        
        if info['exists']:
            info['readable'] = os.access(data_dir, os.R_OK)
            info['writable'] = os.access(data_dir, os.W_OK)
            
            # Calculate directory size
            try:
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(data_dir):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)
                info['size_mb'] = total_size / (1024 * 1024)
            except Exception:
                info['size_mb'] = -1
        else:
            # Try to create directory
            try:
                os.makedirs(data_dir, exist_ok=True)
                info['exists'] = True
                info['readable'] = True
                info['writable'] = True
            except Exception as e:
                info['creation_error'] = str(e)
                
        return info
    
    def get_optimal_device(self) -> torch.device:
        """
        Get the optimal device based on availability and compatibility checks.
        
        Returns:
            torch.device: The recommended device for computation
        """
        if not hasattr(self, 'results') or not self.results:
            self.run_full_environment_check()
            
        compatibility = self.results.get('compatibility', {})
        recommended = compatibility.get('recommended_device', 'cpu')
        
        try:
            import torch
            
            if recommended == 'cuda' and torch.cuda.is_available():
                return torch.device('cuda')
            elif recommended == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
                
        except ImportError:
            return torch.device('cpu')
    
    def setup_device_with_fallback(self) -> Tuple[torch.device, str]:
        """
        Setup device with comprehensive fallback mechanism.
        
        Returns:
            Tuple of (device, status_message)
        """
        try:
            import torch
            
            # First try the shared module if available
            if shared is not None:
                try:
                    device = shared.get_cloud_device(self.config)
                    return device, f"✅ Device configured via shared module: {device}"
                except Exception as e:
                    if self.config.get('verbose', True):
                        print(f"❌ Shared module device setup failed: {e}")
            
            # Fallback to our own device detection
            optimal_device = self.get_optimal_device()
            
            # Test the device with a simple operation
            try:
                test_tensor = torch.randn(2, 2).to(optimal_device)
                _ = test_tensor @ test_tensor.T
                return optimal_device, f"✅ Device tested and confirmed: {optimal_device}"
            except Exception as e:
                # Final fallback to CPU
                cpu_device = torch.device('cpu')
                return cpu_device, f"⚠️ Fallback to CPU due to device test failure: {e}"
                
        except ImportError:
            return torch.device('cpu'), "⚠️ PyTorch not available, using CPU placeholder"
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> list:
        """Generate recommendations based on check results."""
        recommendations = []
        
        # Device recommendations
        device_info = results.get('device_info', {})
        compatibility = results.get('compatibility', {})
        
        if device_info.get('cuda', {}).get('available') and not compatibility.get('cuda_operations'):
            recommendations.append("⚠️ CUDA available but operations failing - check CUDA installation")
            
        if device_info.get('mps', {}).get('available') and not compatibility.get('mps_grid_sampler'):
            recommendations.append("⚠️ MPS available but grid_sample operations failing - use CPU for adversarial attacks")
            
        # Data directory recommendations
        data_info = results.get('data_setup', {})
        if not data_info.get('exists'):
            recommendations.append("📁 Data directory doesn't exist - will be created automatically")
        elif not data_info.get('writable'):
            recommendations.append("❌ Data directory not writable - check permissions")
            
        # PyTorch recommendations
        pytorch_info = results.get('pytorch_info', {})
        if 'error' in pytorch_info:
            recommendations.append("❌ PyTorch not available - install PyTorch for full functionality")
        elif not pytorch_info.get('cuda_available') and device_info.get('cuda', {}).get('available'):
            recommendations.append("⚠️ CUDA detected but PyTorch CUDA not available - reinstall PyTorch with CUDA support")
            
        return recommendations
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a comprehensive summary of environment check results."""
        print("\n" + "="*60)
        print("🔍 COMPREHENSIVE ENVIRONMENT CHECK SUMMARY")
        print("="*60)
        
        # System info
        system = results.get('system_info', {})
        print(f"💻 System: {system.get('system', 'Unknown')} {system.get('machine', '')}")
        
        # Python info
        python = results.get('python_info', {})
        print(f"🐍 Python: {python.get('python_version', 'Unknown')}")
        
        # PyTorch info
        pytorch = results.get('pytorch_info', {})
        if 'error' not in pytorch:
            print(f"🔥 PyTorch: {pytorch.get('version', 'Unknown')}")
            print(f"🖥️ CUDA Available: {pytorch.get('cuda_available', False)}")
            print(f"🍎 MPS Available: {pytorch.get('mps_available', False)}")
        else:
            print("❌ PyTorch: Not available")
        
        # Device recommendations
        compatibility = results.get('compatibility', {})
        recommended = compatibility.get('recommended_device', 'cpu')
        print(f"✅ Recommended Device: {recommended.upper()}")
        
        # Issues
        issues = compatibility.get('issues', [])
        if issues:
            print(f"⚠️ Issues Found: {len(issues)}")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"   • {issue}")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")
        
        print("="*60)


def quick_environment_check(config: Optional[Dict[str, Any]] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Quick environment check function for backward compatibility.
    
    Args:
        config: Configuration dictionary
        verbose: Whether to print results
        
    Returns:
        Dictionary with basic environment information
    """
    if verbose:
        print("🔍 Quick environment check...")
    
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pytorch_version = "Not installed"
        cuda_available = False
    
    # Check data directory
    data_dir = config.get('data_dir', '../data') if config else '../data'
    data_exists = os.path.exists(data_dir)
    
    results = {
        'pytorch_version': pytorch_version,
        'cuda_available': cuda_available,
        'data_dir': data_dir,
        'data_exists': data_exists
    }
    
    if verbose:
        print(f"✅ PyTorch: {pytorch_version}")
        print(f"🖥️ CUDA available: {cuda_available}")
        print(f"📁 Data dir: {data_dir} {'(exists)' if data_exists else '(will create)'}")
        print("✅ Basic environment OK" if pytorch_version != "Not installed" else "⚠️ PyTorch not available")
    
    return results


def check_device_compatibility_for_attacks(device: torch.device, verbose: bool = True) -> Dict[str, Any]:
    """
    Check device compatibility specifically for adversarial attacks.
    
    Args:
        device: PyTorch device to test
        verbose: Whether to print results
        
    Returns:
        Dictionary with compatibility results
    """
    if verbose:
        print("🔧 Checking device compatibility for adversarial attacks...")
    
    results = {
        'device': str(device),
        'supports_attacks': True,
        'issues': [],
        'recommended_device': str(device)
    }
    
    try:
        import torch
        import torch.nn.functional as F
        
        # Test for MPS grid_sample issue (affects affine transformations)
        if str(device) == 'mps':
            if verbose:
                print("⚠️ MPS device detected - testing grid sampler compatibility...")
            
            try:
                # Test the specific operation that fails on MPS
                test_input = torch.randn(1, 1, 4, 4, device=device)
                grid = torch.randn(1, 2, 2, 2, device=device)
                _ = F.grid_sample(test_input, grid, align_corners=False)
                
                if verbose:
                    print("✅ MPS device supports adversarial attacks")
            except Exception as e:
                results['supports_attacks'] = False
                results['issues'].append(f"grid_sample not supported: {e}")
                results['recommended_device'] = 'cpu'
                
                if verbose:
                    print("❌ MPS device doesn't support grid_sampler_2d_backward")
                    print("🔄 Switching to CPU for adversarial attacks...")
        
        # Test basic tensor operations
        try:
            test_tensor = torch.randn(10, 10, device=device)
            _ = test_tensor @ test_tensor.T
        except Exception as e:
            results['supports_attacks'] = False
            results['issues'].append(f"Basic operations failed: {e}")
            results['recommended_device'] = 'cpu'
            
    except ImportError:
        results['supports_attacks'] = False
        results['issues'].append("PyTorch not available")
        results['recommended_device'] = 'cpu'
    
    if verbose:
        status = "✅" if results['supports_attacks'] else "❌"
        print(f"{status} Device compatibility check completed!")
        if not results['supports_attacks']:
            print(f"🔄 Recommended device: {results['recommended_device']}")
    
    return results


# Convenience function for getting attack-compatible device
def get_attack_compatible_device(config: Optional[Dict[str, Any]] = None) -> torch.device:
    """
    Get a device that's compatible with adversarial attacks.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        torch.device that supports adversarial operations
    """
    checker = EnvironmentChecker(config)
    device = checker.get_optimal_device()
    
    # Check if this device supports attacks
    compatibility = check_device_compatibility_for_attacks(device, verbose=False)
    
    if compatibility['supports_attacks']:
        return device
    else:
        # Fall back to recommended device
        recommended = compatibility['recommended_device']
        return torch.device(recommended)


# Legacy function names for backward compatibility
def cuda_availability_check():
    """Legacy function - use EnvironmentChecker instead."""
    warnings.warn("cuda_availability_check is deprecated, use EnvironmentChecker.check_device_availability", 
                  DeprecationWarning)
    return quick_environment_check()


def mps_compatibility_test(device):
    """Legacy function - use check_device_compatibility_for_attacks instead."""
    warnings.warn("mps_compatibility_test is deprecated, use check_device_compatibility_for_attacks", 
                  DeprecationWarning)
    return check_device_compatibility_for_attacks(device)


if __name__ == "__main__":
    # Demo of the environment checker
    print("🚀 Environment Checker Demo")
    print("="*50)
    
    # Create checker with default config
    checker = EnvironmentChecker()
    
    # Run full check
    results = checker.run_full_environment_check()
    
    # Get optimal device
    device = checker.get_optimal_device()
    print(f"\n🎯 Optimal device for this system: {device}")
    
    # Test device compatibility for attacks
    attack_compat = check_device_compatibility_for_attacks(device)
    
    print(f"\n🛡️ Attack compatibility: {'✅ Supported' if attack_compat['supports_attacks'] else '❌ Not supported'}")
