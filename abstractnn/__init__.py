"""
abstractNN - Formal Verification of Neural Networks
====================================================

A Python library for sound verification of neural networks using
abstract interpretation and affine arithmetic.

Basic Usage:
    >>> from abstractnn import AffineEngine, BoundPropagator, ONNXParser
    >>> parser = ONNXParser('model.onnx')
    >>> layers = parser.parse()
    >>> engine = AffineEngine()
    >>> propagator = BoundPropagator(engine)
    >>> # Verify network...

Modules:
    affine_engine: Affine expression management
    bound_propagator: Layer-by-layer bound propagation
    relaxer: Non-linear activation relaxations
    onnx_parser: ONNX model parsing
    partial_evaluator: Partial network verification
    soundness_checker: Soundness validation

For more information, see the documentation at:
https://abstractnn.readthedocs.io
"""

__version__ = "0.1.0"
__author__ = "Guillaume Berthelot"
__email__ = "contact@flyworthi.ai"
__license__ = "MIT"

# Import main classes for convenient access
from .affine_engine import AffineExpressionEngine, AffineExpression
from .bound_propagator import BoundPropagator
from .relaxer import NonLinearRelaxer
from .onnx_parser import ONNXParser
from .partial_evaluator import (
    PartialNetworkEvaluator,
    ONNXPartialEvaluator,
    verify_partial_soundness,
    quick_soundness_check,
)
from .soundness_checker import SoundnessChecker

# Convenient aliases
AffineEngine = AffineExpressionEngine
Relaxer = NonLinearRelaxer
Parser = ONNXParser

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Core classes
    "AffineExpressionEngine",
    "AffineExpression",
    "BoundPropagator",
    "NonLinearRelaxer",
    "ONNXParser",
    "PartialNetworkEvaluator",
    "ONNXPartialEvaluator",
    "SoundnessChecker",
    # Aliases
    "AffineEngine",
    "Relaxer",
    "Parser",
    # Functions
    "verify_partial_soundness",
    "quick_soundness_check",
]

# Submodule imports for namespace access
from . import affine_engine
from . import bound_propagator
from . import relaxer
from . import onnx_parser
from . import partial_evaluator
from . import soundness_checker
