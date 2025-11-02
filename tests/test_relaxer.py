"""
Tests unitaires pour les relaxations d'activations
"""

import unittest
import numpy as np
from abstractnn.affine_engine import AffineExpression
from abstractnn.relaxer import NonLinearRelaxer


class TestNonLinearRelaxer(unittest.TestCase):
    
    def test_relu_always_active(self):
        """Test ReLU quand l'entrée est toujours positive"""
        expr = AffineExpression(2.0, {0: 1.0}, {0: (1.0, 3.0)})
        relaxer = NonLinearRelaxer()
        
        result = relaxer.relu_relaxation(expr)
        
        # Devrait retourner l'expression inchangée
        self.assertEqual(result.constant, expr.constant)
        self.assertEqual(result.coefficients, expr.coefficients)
    
    def test_relu_always_inactive(self):
        """Test ReLU quand l'entrée est toujours négative"""
        expr = AffineExpression(-2.0, {0: 1.0}, {0: (-3.0, -1.0)})
        relaxer = NonLinearRelaxer()
        
        result = relaxer.relu_relaxation(expr)
        
        # Devrait retourner zéro
        lower, upper = result.get_bounds()
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 0.0)
    
    def test_relu_ambiguous(self):
        """Test ReLU avec relaxation linéaire (cas ambigu)"""
        expr = AffineExpression(0.0, {0: 1.0}, {0: (-1.0, 1.0)})
        relaxer = NonLinearRelaxer()
        
        result = relaxer.relu_relaxation(expr, 'linear')
        
        # Vérifie que les bornes sont correctes
        lower, upper = result.get_bounds()
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)


if __name__ == '__main__':
    unittest.main()
