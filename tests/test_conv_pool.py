"""
Tests unitaires pour les opérations Conv2D et MaxPool2D
"""

import unittest
import numpy as np
from modules.affine_engine import AffineExpression, AffineExpressionEngine


class TestConv2D(unittest.TestCase):
    
    def test_simple_conv2d(self):
        """Test d'une convolution 2D simple"""
        # Crée une petite image 1x1x3x3 (batch, channels, height, width)
        input_data = np.array([[[[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0]]]])
        noise = 0.1
        
        engine = AffineExpressionEngine()
        expressions = engine.create_input_expressions(input_data, noise)
        
        # Noyau de convolution 1x1x2x2 (out_channels, in_channels, kH, kW)
        weight = np.array([[[[1.0, 0.0],
                            [0.0, 1.0]]]])
        bias = np.array([0.0])
        
        input_shape = (1, 1, 3, 3)
        
        result, output_shape = engine.conv2d_layer(
            expressions,
            weight,
            bias,
            input_shape=input_shape,
            stride=(1, 1),
            padding=(0, 0)
        )
        
        # Sortie devrait être de forme (1, 1, 2, 2)
        self.assertEqual(output_shape, (1, 1, 2, 2))
        self.assertEqual(len(result), 4)
        
        # Vérifie la première sortie: 1*1 + 1*5 = 6
        lower, upper = result[0].get_bounds()
        self.assertAlmostEqual((lower + upper) / 2, 6.0, delta=0.3)
    
    def test_conv2d_with_stride(self):
        """Test de convolution avec stride"""
        input_data = np.ones((1, 1, 4, 4))
        noise = 0.0
        
        engine = AffineExpressionEngine()
        expressions = engine.create_input_expressions(input_data, noise)
        
        weight = np.ones((1, 1, 2, 2))
        
        result, output_shape = engine.conv2d_layer(
            expressions,
            weight,
            input_shape=(1, 1, 4, 4),
            stride=(2, 2)
        )
        
        # Avec stride=2, sortie devrait être (1, 1, 2, 2)
        self.assertEqual(output_shape, (1, 1, 2, 2))


class TestMaxPool2D(unittest.TestCase):
    
    def test_simple_maxpool(self):
        """Test de MaxPool2D simple"""
        # Crée une image 1x1x4x4
        input_data = np.array([[[[1.0, 2.0, 3.0, 4.0],
                                [5.0, 6.0, 7.0, 8.0],
                                [9.0, 10.0, 11.0, 12.0],
                                [13.0, 14.0, 15.0, 16.0]]]])
        noise = 0.1
        
        engine = AffineExpressionEngine()
        expressions = engine.create_input_expressions(input_data, noise)
        
        input_shape = (1, 1, 4, 4)
        
        result, output_shape = engine.maxpool2d_layer(
            expressions,
            input_shape=input_shape,
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        
        # Sortie devrait être (1, 1, 2, 2)
        self.assertEqual(output_shape, (1, 1, 2, 2))
        self.assertEqual(len(result), 4)
        
        # La première fenêtre contient [1, 2, 5, 6], max ≈ 6
        lower, upper = result[0].get_bounds()
        self.assertGreaterEqual(upper, 5.9)  # Devrait être proche de 6
    
    def test_maxpool_with_padding(self):
        """Test de MaxPool avec padding"""
        input_data = np.ones((1, 1, 3, 3))
        noise = 0.0
        
        engine = AffineExpressionEngine()
        expressions = engine.create_input_expressions(input_data, noise)
        
        result, output_shape = engine.maxpool2d_layer(
            expressions,
            input_shape=(1, 1, 3, 3),
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1)
        )
        
        # Avec padding=1, sortie devrait être (1, 1, 4, 4)
        self.assertEqual(output_shape, (1, 1, 4, 4))


class TestAvgPool2D(unittest.TestCase):
    
    def test_simple_avgpool(self):
        """Test de AvgPool2D simple"""
        # Crée une image uniforme
        input_data = np.full((1, 1, 4, 4), 2.0)
        noise = 0.1
        
        engine = AffineExpressionEngine()
        expressions = engine.create_input_expressions(input_data, noise)
        
        result, output_shape = engine.avgpool2d_layer(
            expressions,
            input_shape=(1, 1, 4, 4),
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        
        # Sortie devrait être (1, 1, 2, 2)
        self.assertEqual(output_shape, (1, 1, 2, 2))
        
        # Moyenne de valeurs autour de 2.0 devrait donner ~2.0
        for expr in result:
            lower, upper = expr.get_bounds()
            self.assertAlmostEqual((lower + upper) / 2, 2.0, delta=0.3)


if __name__ == '__main__':
    unittest.main()
