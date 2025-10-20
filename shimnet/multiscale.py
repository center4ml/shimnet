import torch
import torch.nn as nn
import math

def beta_from_H(H, target_value=0.5):
    """
    Compute beta for a given H such that the sigmoid function maps H to target_value.
    """
    return - math.log(2/(1+target_value) -1) / H

def compute_betas_and_H_values(H_0, lambda_scale, k, target_value=0.5):
    """
    Compute beta and H values for all scales.
    
    Parameters:
    -----------
    H_0 : float
        Base length scale
    lambda_scale : float
        Scaling factor for successive length scales
    k : int
        Number of scales
    target_value : float
        Target value for sigmoid at each H
    
    Returns:
    --------
    betas : torch.Tensor
        Beta values for all scales, shape (k,)
    H_values : torch.Tensor
        H values for all scales, shape (k,)
    """
    betas = []
    H_values = []
    for i in range(k):
        H_i = H_0 * (lambda_scale ** i)
        beta_i = beta_from_H(H_i, target_value)
        betas.append(beta_i)
        H_values.append(H_i)
    
    return torch.tensor(betas, dtype=torch.float32), torch.tensor(H_values, dtype=torch.float32)

class MultiscaleFeatureExtractor(nn.Module):
    """
    Extracts multiscale features from 1D signals using sigmoid-based transformations.
    
    Parameters:
    -----------
    H_0 : float
        Base length scale
    lambda_scale : float
        Scaling factor for successive length scales (H_{i+1} = lambda * H_i)
    k : int
        Number of feature scales to extract
    target_value : float
        Target value for sigmoid at H_0 (default: 0.5)
    concatenate_original : bool
        Whether to concatenate original signal as first channel
    """
    def __init__(self, H_0, lambda_scale, k, target_value=0.5, concatenate_original=False):
        super().__init__()
        self.H_0 = H_0
        self.lambda_scale = lambda_scale
        self.k = k
        self.target_value = target_value
        self.concatenate_original = concatenate_original
        
        # Compute all betas and H values
        betas, H_values = compute_betas_and_H_values(H_0, lambda_scale, k, target_value)
        
        # Register as buffers (will move with model to GPU/CPU)
        self.register_buffer('betas', betas)
        self.register_buffer('H_values', H_values)
    
    def extract_features(self, x):
        """
        Extract multiscale features from input signal batch.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input signal batch of shape (batch_size, 1, length)
        
        Returns:
        --------
        features : torch.Tensor
            Multiscale features of shape (batch_size, k, length) or (batch_size, k+1, length)
        """
        # Ensure input has correct shape (batch_size, 1, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, length) -> (batch_size, 1, length)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (length,) -> (1, 1, length)
        
        # Vectorized operation: broadcast betas over batch and length dimensions
        # self.betas: (k,) -> (1, k, 1)
        # x: (batch_size, 1, length)
        # Result: (batch_size, k, length)
        features = 2 * torch.sigmoid(self.betas.view(1, -1, 1) * x) - 1
        
        if self.concatenate_original:
            features = torch.cat([x, features], dim=1)  # (batch_size, 1+k, length)
        
        return features
    
    def forward(self, x):
        """Forward pass (standard nn.Module convention)"""
        return self.extract_features(x)
    
    def __call__(self, x):
        """Alias for forward"""
        return self.forward(x)
    
    def __repr__(self):
        info = f"MultiscaleFeatureExtractor(\n"
        info += f"  H_0={self.H_0:.4f}, lambda={self.lambda_scale:.4f}, k={self.k}\n"
        info += f"  target_value={self.target_value}\n"
        info += f"  concatenate_original={self.concatenate_original}\n"
        info += f"  Length scales: {[f'{h:.4f}' for h in self.H_values.tolist()]}\n"
        info += f"  Beta values: {[f'{b:.4f}' for b in self.betas.tolist()]}\n"
        info += f")"
        return info