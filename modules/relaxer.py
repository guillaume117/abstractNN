"""
Relaxation des fonctions d'activation non-linéaires
"""

import numpy as np
from typing import Tuple
from .affine_engine import AffineExpression


class NonLinearRelaxer:
    """Gère les relaxations convexes des activations non-linéaires"""
    
    @staticmethod
    def relu_relaxation(expr: AffineExpression, 
                       relaxation_type: str = 'linear') -> AffineExpression:
        """
        Relaxation de ReLU: z = max(0, y)
        
        Si y ∈ [l, u]:
        - Si u ≤ 0 → z = 0
        - Si l ≥ 0 → z = y
        - Sinon → relaxation linéaire: z ≤ (u/(u-l)) * (y - l)
        """
        l, u = expr.get_bounds()
        
        # Cas où l'activation est toujours désactivée
        if u <= 0:
            return AffineExpression(0.0, {}, expr.bounds)
        
        # Cas où l'activation est toujours active
        if l >= 0:
            return expr
        
        # Cas ambigu: relaxation linéaire
        if relaxation_type == 'linear':
            # z ≥ 0, z ≥ y, z ≤ (u/(u-l)) * (y - l)
            slope = u / (u - l)
            
            new_coeffs = {idx: coeff * slope 
                         for idx, coeff in expr.coefficients.items()}
            new_constant = (expr.constant - l) * slope
            
            return AffineExpression(new_constant, new_coeffs, expr.bounds)
        
        return expr
    
    @staticmethod
    def sigmoid_relaxation(expr: AffineExpression) -> AffineExpression:
        """Relaxation de Sigmoid (approximation linéaire)"""
        l, u = expr.get_bounds()
        
        # Approximation linéaire au point milieu
        mid = (l + u) / 2
        sig_mid = 1 / (1 + np.exp(-mid))
        slope = sig_mid * (1 - sig_mid)
        
        new_coeffs = {idx: coeff * slope 
                     for idx, coeff in expr.coefficients.items()}
        new_constant = sig_mid - slope * mid + expr.constant * slope
        
        return AffineExpression(new_constant, new_coeffs, expr.bounds)
    
    @staticmethod
    def tanh_relaxation(expr: AffineExpression) -> AffineExpression:
        """Relaxation de Tanh (approximation linéaire)"""
        l, u = expr.get_bounds()
        
        # Approximation linéaire
        mid = (l + u) / 2
        tanh_mid = np.tanh(mid)
        slope = 1 - tanh_mid ** 2
        
        new_coeffs = {idx: coeff * slope 
                     for idx, coeff in expr.coefficients.items()}
        new_constant = tanh_mid - slope * mid + expr.constant * slope
        
        return AffineExpression(new_constant, new_coeffs, expr.bounds)
