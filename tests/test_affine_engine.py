"""
Tests unitaires pour le moteur d'expressions affines
"""

import unittest
import numpy as np
from abstractnn.affine_engine import AffineExpression, AffineExpressionEngine


class TestAffineExpression(unittest.TestCase):
    
    def test_simple_bounds(self):
        """Test du calcul de bornes simples"""
        expr = AffineExpression(
            constant=1.0,
            coefficients={0: 2.0},
            bounds={0: (0.0, 1.0)}
        )
        lower, upper = expr.get_bounds()
        self.assertEqual(lower, 1.0)
        self.assertEqual(upper, 3.0)
    
    def test_addition(self):
        """Test de l'addition d'expressions"""
        expr1 = AffineExpression(1.0, {0: 2.0}, {0: (0.0, 1.0)})
        expr2 = AffineExpression(0.5, {0: 1.0}, {0: (0.0, 1.0)})
        
        result = expr1 + expr2
        self.assertEqual(result.constant, 1.5)
        self.assertEqual(result.coefficients[0], 3.0)
    
    def test_multiplication(self):
        """Test de la multiplication par un scalaire"""
        expr = AffineExpression(1.0, {0: 2.0}, {0: (0.0, 1.0)})
        result = expr * 2.0
        
        self.assertEqual(result.constant, 2.0)
        self.assertEqual(result.coefficients[0], 4.0)


class TestAffineExpressionEngine(unittest.TestCase):
    
    def test_create_input_expressions(self):
        """Test de la création des expressions d'entrée"""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        noise = 0.1
        
        engine = AffineExpressionEngine()
        expressions = engine.create_input_expressions(input_data, noise)
        
        self.assertEqual(len(expressions), 4)
        
        # Vérifie les bornes du premier pixel (valeur 1.0 ± 0.1)
        lower, upper = expressions[0].get_bounds()
        self.assertAlmostEqual(lower, 0.9)
        self.assertAlmostEqual(upper, 1.1)
        
        # Vérifie les bornes du dernier pixel (valeur 4.0 ± 0.1)
        lower, upper = expressions[3].get_bounds()
        self.assertAlmostEqual(lower, 3.9)
        self.assertAlmostEqual(upper, 4.1)
    
    def test_linear_layer(self):
        """Test de la propagation à travers une couche linéaire"""
        # Crée 2 expressions simples représentant des pixels avec bornes
        expr1 = AffineExpression(0.0, {0: 1.0}, {0: (0.9, 1.1)})  # pixel ~1.0
        expr2 = AffineExpression(0.0, {1: 1.0}, {1: (1.9, 2.1)})  # pixel ~2.0
        
        weight = np.array([[1.0, 2.0]])
        bias = np.array([0.5])
        
        engine = AffineExpressionEngine()
        result = engine.linear_layer([expr1, expr2], weight, bias)
        
        self.assertEqual(len(result), 1)
        
        # La sortie devrait être: 0.5 + 1.0*x0 + 2.0*x1
        # avec x0 ∈ [0.9, 1.1] et x1 ∈ [1.9, 2.1]
        self.assertAlmostEqual(result[0].constant, 0.5)
        self.assertAlmostEqual(result[0].coefficients[0], 1.0)
        self.assertAlmostEqual(result[0].coefficients[1], 2.0)
        
        # Bornes: min = 0.5 + 1.0*0.9 + 2.0*1.9 = 0.5 + 0.9 + 3.8 = 5.2
        #         max = 0.5 + 1.0*1.1 + 2.0*2.1 = 0.5 + 1.1 + 4.2 = 5.8
        lower, upper = result[0].get_bounds()
        self.assertAlmostEqual(lower, 5.2)
        self.assertAlmostEqual(upper, 5.8)


if __name__ == '__main__':
    unittest.main()
