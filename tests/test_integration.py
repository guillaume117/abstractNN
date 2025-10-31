"""
Tests d'intégration
"""

import unittest
import numpy as np
from modules.affine_engine import AffineExpressionEngine
from modules.bound_propagator import BoundPropagator
from modules.result_aggregator import ResultAggregator


class TestIntegration(unittest.TestCase):
    
    def test_simple_network(self):
        """Test sur un réseau simple (Linear + ReLU + Linear)"""
        # Crée une entrée simple
        input_data = np.array([1.0, 2.0])
        noise = 0.1
        
        engine = AffineExpressionEngine()
        expressions = engine.create_input_expressions(input_data, noise)
        
        # Définit un réseau simple
        layers = [
            {
                'type': 'MatMul',
                'name': 'fc1',
                'inputs': ['input'],
                'outputs': ['fc1_out'],
                'attributes': {},
                'weights': np.array([[1.0, 1.0], [1.0, -1.0]]),
                'bias': np.array([0.0, 0.0])
            },
            {
                'type': 'Relu',
                'name': 'relu1',
                'inputs': ['fc1_out'],
                'outputs': ['relu1_out'],
                'attributes': {}
            },
            {
                'type': 'MatMul',
                'name': 'fc2',
                'inputs': ['relu1_out'],
                'outputs': ['output'],
                'attributes': {},
                'weights': np.array([[1.0, 0.0], [0.0, 1.0]]),
                'bias': np.array([0.0, 0.0])
            }
        ]
        
        # Propage
        propagator = BoundPropagator(relaxation_type='linear')
        output_expr = propagator.propagate(expressions, layers)
        
        # Agrège
        aggregator = ResultAggregator()
        bounds = aggregator.compute_class_bounds(output_expr)
        
        # Vérifie qu'on a bien 2 classes avec des bornes
        self.assertEqual(len(bounds), 2)
        for class_name, (lower, upper) in bounds.items():
            self.assertLessEqual(lower, upper)
    
    def test_robust_class_detection(self):
        """Test de la détection de classe robuste"""
        aggregator = ResultAggregator()
        
        # Cas avec une classe clairement robuste
        bounds = {
            'class_0': (5.0, 6.0),
            'class_1': (1.0, 2.0),
            'class_2': (0.5, 1.5)
        }
        
        robust = aggregator.find_robust_class(bounds)
        self.assertEqual(robust, 'class_0')
        
        # Cas sans classe robuste
        bounds_ambiguous = {
            'class_0': (1.0, 3.0),
            'class_1': (2.0, 4.0)
        }
        
        robust = aggregator.find_robust_class(bounds_ambiguous)
        self.assertIsNone(robust)


if __name__ == '__main__':
    unittest.main()
