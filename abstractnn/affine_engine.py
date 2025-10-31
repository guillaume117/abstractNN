"""
Moteur d'expressions affines pour la propagation formelle
"""

import numpy as np
from typing import Dict, Tuple, List


class AffineExpression:
    """
    Représente une expression affine : y = a0 + sum(aj * xj)
    avec xj ∈ [lj, uj]
    """
    
    def __init__(self, constant: float, coefficients: Dict[int, float], 
                 bounds: Dict[int, Tuple[float, float]]):
        self.constant = constant
        self.coefficients = coefficients  # {index: coefficient}
        self.bounds = bounds  # {index: (lower, upper)}
    
    def get_bounds(self) -> Tuple[float, float]:
        """Calcule les bornes [min, max] de l'expression affine"""
        lower = self.constant
        upper = self.constant
        
        for idx, coeff in self.coefficients.items():
            l_bound, u_bound = self.bounds[idx]
            if coeff >= 0:
                lower += coeff * l_bound
                upper += coeff * u_bound
            else:
                lower += coeff * u_bound
                upper += coeff * l_bound
        
        return (lower, upper)
    
    def __add__(self, other):
        """Addition de deux expressions affines"""
        if isinstance(other, (int, float)):
            return AffineExpression(
                self.constant + other,
                self.coefficients.copy(),
                self.bounds.copy()
            )
        
        new_coeffs = self.coefficients.copy()
        for idx, coeff in other.coefficients.items():
            new_coeffs[idx] = new_coeffs.get(idx, 0.0) + coeff
        
        return AffineExpression(
            self.constant + other.constant,
            new_coeffs,
            {**self.bounds, **other.bounds}
        )
    
    def __mul__(self, scalar):
        """Multiplication par un scalaire"""
        return AffineExpression(
            self.constant * scalar,
            {idx: coeff * scalar for idx, coeff in self.coefficients.items()},
            self.bounds.copy()
        )
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)


class AffineExpressionEngine:
    """Moteur de manipulation d'expressions affines"""
    
    @staticmethod
    def create_input_expressions(input_data: np.ndarray, 
                                 noise_level: float) -> List[AffineExpression]:
        """
        Crée les expressions affines initiales pour l'entrée avec bruit
        Chaque pixel x_i devient: x_i = 0 + 1*ε_i où ε_i ∈ [value-noise, value+noise]
        """
        flat_input = input_data.flatten()
        expressions = []
        
        for idx, value in enumerate(flat_input):
            # Chaque pixel devient une variable bruitée: 0 + 1*ε_i
            # avec ε_i ∈ [value - noise, value + noise]
            expressions.append(AffineExpression(
                constant=0.0,  # Pas de partie constante
                coefficients={idx: 1.0},  # Coefficient unitaire pour la variable
                bounds={idx: (value - noise_level, value + noise_level)}
            ))
        
        return expressions
    
    @staticmethod
    def linear_layer(expressions: List[AffineExpression], 
                    weight: np.ndarray, 
                    bias: np.ndarray = None) -> List[AffineExpression]:
        """Propage à travers une couche linéaire (MatMul + bias)"""
        n_outputs = weight.shape[0]
        result = []
        
        for i in range(n_outputs):
            # y_i = sum(w_ij * x_j) + b_i
            constant = 0.0
            coefficients = {}
            bounds = {}
            
            for j, expr in enumerate(expressions):
                w = weight[i, j]
                constant += w * expr.constant
                
                for idx, coeff in expr.coefficients.items():
                    coefficients[idx] = coefficients.get(idx, 0.0) + w * coeff
                    if idx in expr.bounds:
                        bounds[idx] = expr.bounds[idx]
            
            if bias is not None:
                constant += bias[i]
            
            result.append(AffineExpression(constant, coefficients, bounds))
        
        return result
    
    @staticmethod
    def conv2d_layer(expressions: List[AffineExpression],
                     weight: np.ndarray,
                     bias: np.ndarray = None,
                     input_shape: Tuple[int, int, int, int] = None,
                     stride: Tuple[int, int] = (1, 1),
                     padding: Tuple[int, int] = (0, 0),
                     dilation: Tuple[int, int] = (1, 1)) -> Tuple[List[AffineExpression], Tuple[int, int, int, int]]:
        """
        Propage à travers une couche convolutionnelle 2D
        
        Args:
            expressions: Liste d'expressions affines (input flattened)
            weight: Poids de convolution de forme (out_channels, in_channels, kH, kW)
            bias: Biais de forme (out_channels,)
            input_shape: (batch, in_channels, height, width)
            stride: (stride_h, stride_w)
            padding: (pad_h, pad_w)
            dilation: (dilation_h, dilation_w)
        
        Returns:
            (expressions de sortie, output_shape)
        """
        if input_shape is None:
            # Essaie de déduire la forme depuis les expressions
            raise ValueError("input_shape doit être spécifié pour Conv2D")
        
        batch, in_channels, in_h, in_w = input_shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation
        
        # Calcule les dimensions de sortie
        out_h = (in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
        out_w = (in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
        
        # Reshape les expressions en tenseur 4D
        input_tensor = np.array(expressions).reshape(batch, in_channels, in_h, in_w)
        
        result = []
        
        # Pour chaque position de sortie
        for b in range(batch):
            for oc in range(out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        # Calcule la position de départ dans l'entrée
                        h_start = oh * stride_h - pad_h
                        w_start = ow * stride_w - pad_w
                        
                        # Initialise l'expression de sortie
                        constant = 0.0
                        coefficients = {}
                        bounds = {}
                        
                        # Applique le noyau de convolution
                        for ic in range(in_channels):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    h_pos = h_start + kh * dil_h
                                    w_pos = w_start + kw * dil_w
                                    
                                    # Vérifie les limites (padding implicite = 0)
                                    if 0 <= h_pos < in_h and 0 <= w_pos < in_w:
                                        input_idx = b * (in_channels * in_h * in_w) + \
                                                   ic * (in_h * in_w) + \
                                                   h_pos * in_w + w_pos
                                        
                                        expr = expressions[input_idx]
                                        w_val = weight[oc, ic, kh, kw]
                                        
                                        constant += w_val * expr.constant
                                        
                                        for idx, coeff in expr.coefficients.items():
                                            coefficients[idx] = coefficients.get(idx, 0.0) + w_val * coeff
                                            if idx in expr.bounds:
                                                bounds[idx] = expr.bounds[idx]
                        
                        # Ajoute le biais
                        if bias is not None:
                            constant += bias[oc]
                        
                        result.append(AffineExpression(constant, coefficients, bounds))
        
        output_shape = (batch, out_channels, out_h, out_w)
        return result, output_shape
    
    @staticmethod
    def maxpool2d_layer(expressions: List[AffineExpression],
                       input_shape: Tuple[int, int, int, int],
                       kernel_size: Tuple[int, int],
                       stride: Tuple[int, int] = None,
                       padding: Tuple[int, int] = (0, 0)) -> Tuple[List[AffineExpression], Tuple[int, int, int, int]]:
        """
        Propage à travers une couche MaxPool2D
        
        Pour MaxPool, on utilise une relaxation: on prend l'expression avec la borne supérieure maximale
        (approximation conservative)
        
        Args:
            expressions: Liste d'expressions affines
            input_shape: (batch, channels, height, width)
            kernel_size: (pool_h, pool_w)
            stride: (stride_h, stride_w), par défaut = kernel_size
            padding: (pad_h, pad_w)
        
        Returns:
            (expressions de sortie, output_shape)
        """
        if stride is None:
            stride = kernel_size
        
        batch, channels, in_h, in_w = input_shape
        pool_h, pool_w = kernel_size
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        
        # Calcule les dimensions de sortie
        out_h = (in_h + 2 * pad_h - pool_h) // stride_h + 1
        out_w = (in_w + 2 * pad_w - pool_w) // stride_w + 1
        
        result = []
        
        # Pour chaque position de sortie
        for b in range(batch):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * stride_h - pad_h
                        w_start = ow * stride_w - pad_w
                        
                        # Collecte toutes les expressions dans la fenêtre de pooling
                        pool_expressions = []
                        
                        for ph in range(pool_h):
                            for pw in range(pool_w):
                                h_pos = h_start + ph
                                w_pos = w_start + pw
                                
                                # Vérifie les limites
                                if 0 <= h_pos < in_h and 0 <= w_pos < in_w:
                                    input_idx = b * (channels * in_h * in_w) + \
                                               c * (in_h * in_w) + \
                                               h_pos * in_w + w_pos
                                    pool_expressions.append(expressions[input_idx])
                        
                        # Relaxation MaxPool: prend l'union des bornes
                        # Pour chaque variable de bruit, on prend le max des bornes supérieures
                        if pool_expressions:
                            result_expr = AffineExpressionEngine._maxpool_relaxation(pool_expressions)
                            result.append(result_expr)
        
        output_shape = (batch, channels, out_h, out_w)
        return result, output_shape
    
    @staticmethod
    def _maxpool_relaxation(expressions: List[AffineExpression]) -> AffineExpression:
        """
        Relaxation pour MaxPool: calcule une sur-approximation conservative
        
        Pour max(y1, y2, ..., yn), on utilise:
        - Borne inférieure: max des bornes inférieures
        - Borne supérieure: max des bornes supérieures
        
        Pour l'expression affine, on utilise une approximation linéaire basée sur
        les bornes.
        """
        if len(expressions) == 1:
            return expressions[0]
        
        # Calcule les bornes de chaque expression
        bounds_list = [expr.get_bounds() for expr in expressions]
        
        # Trouve l'expression avec la borne supérieure maximale
        max_upper_idx = max(range(len(bounds_list)), key=lambda i: bounds_list[i][1])
        
        # Utilise cette expression comme base (approximation conservative)
        base_expr = expressions[max_upper_idx]
        
        # Ajuste les bornes pour être conservatives
        all_bounds = {}
        for expr in expressions:
            for idx, bound in expr.bounds.items():
                if idx not in all_bounds:
                    all_bounds[idx] = bound
                else:
                    # Union des bornes
                    old_l, old_u = all_bounds[idx]
                    new_l, new_u = bound
                    all_bounds[idx] = (min(old_l, new_l), max(old_u, new_u))
        
        # Crée une nouvelle expression avec les bornes élargies
        return AffineExpression(
            base_expr.constant,
            base_expr.coefficients.copy(),
            all_bounds
        )
    
    @staticmethod
    def avgpool2d_layer(expressions: List[AffineExpression],
                       input_shape: Tuple[int, int, int, int],
                       kernel_size: Tuple[int, int],
                       stride: Tuple[int, int] = None,
                       padding: Tuple[int, int] = (0, 0)) -> Tuple[List[AffineExpression], Tuple[int, int, int, int]]:
        """
        Propage à travers une couche AvgPool2D (opération linéaire)
        
        Args:
            expressions: Liste d'expressions affines
            input_shape: (batch, channels, height, width)
            kernel_size: (pool_h, pool_w)
            stride: (stride_h, stride_w)
            padding: (pad_h, pad_w)
        
        Returns:
            (expressions de sortie, output_shape)
        """
        if stride is None:
            stride = kernel_size
        
        batch, channels, in_h, in_w = input_shape
        pool_h, pool_w = kernel_size
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        
        out_h = (in_h + 2 * pad_h - pool_h) // stride_h + 1
        out_w = (in_w + 2 * pad_w - pool_w) // stride_w + 1
        
        result = []
        
        for b in range(batch):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * stride_h - pad_h
                        w_start = ow * stride_w - pad_w
                        
                        # Initialise l'expression moyenne
                        constant = 0.0
                        coefficients = {}
                        bounds = {}
                        count = 0
                        
                        for ph in range(pool_h):
                            for pw in range(pool_w):
                                h_pos = h_start + ph
                                w_pos = w_start + pw
                                
                                if 0 <= h_pos < in_h and 0 <= w_pos < in_w:
                                    input_idx = b * (channels * in_h * in_w) + \
                                               c * (in_h * in_w) + \
                                               h_pos * in_w + w_pos
                                    
                                    expr = expressions[input_idx]
                                    constant += expr.constant
                                    
                                    for idx, coeff in expr.coefficients.items():
                                        coefficients[idx] = coefficients.get(idx, 0.0) + coeff
                                        if idx in expr.bounds:
                                            bounds[idx] = expr.bounds[idx]
                                    
                                    count += 1
                        
                        # Moyenne
                        if count > 0:
                            constant /= count
                            coefficients = {idx: coeff / count for idx, coeff in coefficients.items()}
                        
                        result.append(AffineExpression(constant, coefficients, bounds))
        
        output_shape = (batch, channels, out_h, out_w)
        return result, output_shape
