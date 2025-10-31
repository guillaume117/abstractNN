"""
Propagateur de bornes à travers le réseau
"""

import numpy as np
import time
from typing import List, Dict, Any
from .affine_engine import AffineExpression, AffineExpressionEngine
from .relaxer import NonLinearRelaxer
from .report_generator import ReportGenerator


class BoundPropagator:
    """Propage les expressions affines à travers le réseau"""
    
    def __init__(self, relaxation_type: str = 'linear', enable_reporting: bool = True):
        self.relaxation_type = relaxation_type
        self.engine = AffineExpressionEngine()
        self.relaxer = NonLinearRelaxer()
        self.intermediate_bounds = []
        self.current_shape = None
        self.enable_reporting = enable_reporting
        self.reporter = ReportGenerator() if enable_reporting else None
    
    def propagate(self, 
                  initial_expressions: List[AffineExpression],
                  layers: List[Dict[str, Any]],
                  input_shape: tuple = None) -> List[AffineExpression]:
        """Propage les expressions à travers toutes les couches"""
        expressions = initial_expressions
        self.current_shape = input_shape
        
        for layer_idx, layer in enumerate(layers):
            layer_start_time = time.time()
            layer_type = layer['type']
            
            # Propage selon le type de couche
            if layer_type in ['MatMul', 'Gemm']:
                expressions = self._propagate_linear(expressions, layer)
            elif layer_type == 'Conv':
                expressions = self._propagate_conv(expressions, layer)
            elif layer_type == 'MaxPool':
                expressions = self._propagate_maxpool(expressions, layer)
            elif layer_type == 'AveragePool':
                expressions = self._propagate_avgpool(expressions, layer)
            elif layer_type == 'Relu':
                expressions = self._propagate_relu(expressions)
            elif layer_type == 'Sigmoid':
                expressions = self._propagate_sigmoid(expressions)
            elif layer_type == 'Tanh':
                expressions = self._propagate_tanh(expressions)
            elif layer_type == 'Flatten':
                expressions = self._propagate_flatten(expressions, layer)
            elif layer_type == 'Reshape':
                expressions = self._propagate_reshape(expressions, layer)
            elif layer_type == 'Add':
                expressions = self._propagate_add(expressions, layer)
            
            layer_time = time.time() - layer_start_time
            
            # Enregistre les bornes intermédiaires
            bounds = [expr.get_bounds() for expr in expressions]
            self.intermediate_bounds.append({
                'layer': layer['name'],
                'type': layer_type,
                'bounds': bounds,
                'shape': self.current_shape
            })
            
            # Génère le rapport si activé
            if self.reporter:
                self.reporter.add_layer_report(expressions, layer, layer_time)
        
        # Génère le résumé final
        if self.reporter:
            self.reporter.generate_summary()
        
        return expressions
    
    def get_report(self) -> ReportGenerator:
        """Retourne le générateur de rapport"""
        return self.reporter
    
    def _propagate_linear(self, expressions: List[AffineExpression],
                         layer: Dict[str, Any]) -> List[AffineExpression]:
        """Propage à travers une couche linéaire"""
        weights = layer.get('weights')
        bias = layer.get('bias')
        
        if weights is not None:
            # Vérifie que les dimensions correspondent
            if len(weights.shape) == 2:
                # Reshape les expressions en vecteur si nécessaire
                if self.current_shape is not None and len(self.current_shape) == 4:
                    # Flatten automatique avant la couche linéaire
                    batch = self.current_shape[0]
                    total = len(expressions) // batch if batch > 0 else len(expressions)
                    self.current_shape = (batch, total)
                
                result = self.engine.linear_layer(expressions, weights, bias)
                
                # Met à jour la forme
                if self.current_shape is not None:
                    self.current_shape = (self.current_shape[0], len(result))
                
                return result
        
        return expressions
    
    def _propagate_conv(self, expressions: List[AffineExpression],
                       layer: Dict[str, Any]) -> List[AffineExpression]:
        """Propage à travers une couche convolutionnelle"""
        weights = layer.get('weights')
        bias = layer.get('bias')
        
        if weights is None:
            print(f"Attention: pas de poids pour la couche {layer['name']}")
            return expressions
        
        # Vérifie que les poids ont la bonne forme
        if len(weights.shape) != 4:
            print(f"Attention: forme de poids incorrecte pour Conv: {weights.shape}")
            return expressions
        
        # Extrait les paramètres de convolution
        attrs = layer.get('attributes', {})
        strides = attrs.get('strides', [1, 1])
        pads = attrs.get('pads', [0, 0, 0, 0])  # [top, left, bottom, right]
        dilations = attrs.get('dilations', [1, 1])
        
        # Convertit pads en (pad_h, pad_w)
        padding = (pads[0], pads[1])
        stride = tuple(strides)
        dilation = tuple(dilations)
        
        if self.current_shape is not None:
            expressions, self.current_shape = self.engine.conv2d_layer(
                expressions, 
                weights, 
                bias,
                input_shape=self.current_shape,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        
        return expressions
    
    def _propagate_maxpool(self, expressions: List[AffineExpression],
                          layer: Dict[str, Any]) -> List[AffineExpression]:
        """Propage à travers MaxPool2D"""
        attrs = layer.get('attributes', {})
        kernel_shape = attrs.get('kernel_shape', [2, 2])
        strides = attrs.get('strides', kernel_shape)
        pads = attrs.get('pads', [0, 0, 0, 0])
        
        padding = (pads[0], pads[1])
        stride = tuple(strides)
        
        if self.current_shape is not None:
            expressions, self.current_shape = self.engine.maxpool2d_layer(
                expressions,
                input_shape=self.current_shape,
                kernel_size=tuple(kernel_shape),
                stride=stride,
                padding=padding
            )
        
        return expressions
    
    def _propagate_avgpool(self, expressions: List[AffineExpression],
                          layer: Dict[str, Any]) -> List[AffineExpression]:
        """Propage à travers AveragePool2D"""
        attrs = layer.get('attributes', {})
        kernel_shape = attrs.get('kernel_shape', [2, 2])
        strides = attrs.get('strides', kernel_shape)
        pads = attrs.get('pads', [0, 0, 0, 0])
        
        padding = (pads[0], pads[1])
        stride = tuple(strides)
        
        if self.current_shape is not None:
            expressions, self.current_shape = self.engine.avgpool2d_layer(
                expressions,
                input_shape=self.current_shape,
                kernel_size=tuple(kernel_shape),
                stride=stride,
                padding=padding
            )
        
        return expressions
    
    def _propagate_relu(self, 
                       expressions: List[AffineExpression]) -> List[AffineExpression]:
        """Propage à travers ReLU"""
        return [self.relaxer.relu_relaxation(expr, self.relaxation_type) 
                for expr in expressions]
    
    def _propagate_sigmoid(self, 
                          expressions: List[AffineExpression]) -> List[AffineExpression]:
        """Propage à travers Sigmoid"""
        return [self.relaxer.sigmoid_relaxation(expr) for expr in expressions]
    
    def _propagate_tanh(self, 
                       expressions: List[AffineExpression]) -> List[AffineExpression]:
        """Propage à travers Tanh"""
        return [self.relaxer.tanh_relaxation(expr) for expr in expressions]
    
    def _propagate_flatten(self, expressions: List[AffineExpression],
                          layer: Dict[str, Any]) -> List[AffineExpression]:
        """Propage à travers Flatten"""
        if self.current_shape is not None:
            batch = self.current_shape[0]
            total = len(expressions) // batch if batch > 0 else len(expressions)
            self.current_shape = (batch, total)
        
        return expressions
    
    def _propagate_reshape(self, expressions: List[AffineExpression],
                          layer: Dict[str, Any]) -> List[AffineExpression]:
        """Propage à travers Reshape"""
        attrs = layer.get('attributes', {})
        new_shape = attrs.get('shape', None)
        
        if new_shape is not None:
            self.current_shape = tuple(new_shape)
        
        return expressions
    
    def _propagate_add(self, expressions: List[AffineExpression],
                      layer: Dict[str, Any]) -> List[AffineExpression]:
        """Propage à travers une addition"""
        # Simplifié - suppose addition élément par élément
        return expressions
