import torch
import math

def beta_from_H(H, target_value=0.5):
    """
    Compute beta for a given H such that the sigmoid function maps H to target_value.
    """
    return - math.log(2/(1+target_value) -1)/ H

class MultiscaleFeatureExtractor:
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
    """
    def __init__(self, H_0, lambda_scale, k, target_value=0.5):
        self.H_0 = H_0
        self.lambda_scale = lambda_scale
        self.k = k
        self.target_value = target_value
        
        self.beta_0 = beta_from_H(H_0, target_value)

        # Compute all betas and H values
        self.betas = []
        self.H_values = []
        for i in range(k):
            H_i = H_0 * (lambda_scale ** i)
            beta_i = beta_from_H(H_i, target_value)
            self.betas.append(beta_i)
            self.H_values.append(H_i)
    
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
            Multiscale features of shape (batch_size, k, length)
        """
        # Ensure input has correct shape (batch_size, 1, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, length) -> (batch_size, 1, length)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (length,) -> (1, 1, length)
        
        # Squeeze out the channel dimension for processing
        x = x.squeeze(1)  # (batch_size, 1, length) -> (batch_size, length)
        
        features = []
        for beta_i in self.betas:
            # Apply: 2*sigmoid(beta_i * x) - 1
            feature_i = 2 * torch.sigmoid(beta_i * x) - 1
            features.append(feature_i)
        
        # Stack along channel dimension: (k, batch_size, length) -> (batch_size, k, length)
        features = torch.stack(features, dim=0)  # (k, batch_size, length)
        features = features.permute(1, 0, 2)  # (batch_size, k, length)
        
        return features
    
    def __call__(self, x):
        """alias for extract_features"""
        return self.extract_features(x)
    
    def __repr__(self):
        info = f"MultiscaleFeatureExtractor(\n"
        info += f"  H_0={self.H_0:.4f}, lambda={self.lambda_scale:.4f}, k={self.k}\n"
        info += f"  beta_0={self.beta_0:.4f}, target_value={self.target_value}\n"
        info += f"  Length scales: {[f'{h:.4f}' for h in self.H_values]}\n"
        info += f"  Beta values: {[f'{b:.4f}' for b in self.betas]}\n"
        info += f")"
        return info