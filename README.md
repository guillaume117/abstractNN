# abstractNN

[![PyPI version](https://badge.fury.io/py/abstractNN.svg)](https://badge.fury.io/py/abstractNN)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/abstractnn/badge/?version=latest)](https://abstractnn.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**abstractNN** is a Python library for **formal verification of neural networks** using abstract interpretation and affine arithmetic. It provides **mathematically sound** guarantees about network behavior under input perturbations.

## 🎯 Key Features

- **Sound Verification**: Mathematically proven bounds on network outputs
- **Affine Arithmetic**: Symbolic tracking of input-output dependencies
- **Multiple Relaxations**: Linear, interval, and hybrid relaxation strategies
- **ONNX Support**: Works with models exported from PyTorch, TensorFlow, etc.
- **Partial Evaluation**: Efficient verification of network sub-regions
- **Soundness Checking**: Monte Carlo validation of formal bounds
- **Production Ready**: Well-tested, documented, and type-hinted

## 📦 Installation

### From PyPI (recommended)

```bash
pip install abstractNN
```

### From source

```bash
git clone https://github.com/guillaume117/abstractNN.git
cd abstractNN
pip install -e .

# Download pre-trained models
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

### With optional dependencies

```bash
# Development tools
pip install abstractNN[dev]

# Documentation building
pip install abstractNN[docs]

# GPU acceleration
pip install abstractNN[gpu]

# Visualization tools
pip install abstractNN[viz]

# All extras
pip install abstractNN[dev,docs,gpu,viz]
```

## 🚀 Quick Start

### Basic Verification

```python
from abstractnn import AffineEngine, BoundPropagator, ONNXParser

# Load your ONNX model
parser = ONNXParser('model.onnx')
layers = parser.parse()

# Create verification engine
engine = AffineEngine()
propagator = BoundPropagator(engine)

# Define perturbed input
import numpy as np
image = np.random.rand(3, 28, 28).astype(np.float32)
noise_level = 0.1  # L∞ perturbation radius

# Create symbolic expressions
input_exprs = engine.create_input_expressions(image, noise_level)

# Propagate through network
output_exprs = propagator.propagate(input_exprs, layers, image.shape)

# Get guaranteed output bounds
bounds = [expr.get_bounds() for expr in output_exprs]
print(f"Output bounds: [{bounds[0][0]:.4f}, {bounds[0][1]:.4f}]")
```

### VGG16 Partial Verification

```python
from abstractnn import verify_partial_soundness

# Verify first 5 layers of VGG16
result = verify_partial_soundness(
    model_path='vgg16.onnx',
    image=test_image,
    noise_level=0.01,
    num_layers=5,
    num_mc_samples=100
)

if result['success']:
    print(f"Sound: {result['soundness_report']['is_sound']}")
    print(f"Coverage: {result['soundness_report']['coverage_ratio']*100:.1f}%")
```

### Command Line Interface

```bash
# Verify a model
abstractnn-verify --model model.onnx --image test.npy --epsilon 0.01

# Get library info
abstractnn-info
```

## ⚠️ Large Model Files

**Note**: Pre-trained model files (like VGG16) are **NOT** included in the repository due to their large size (500+ MB).

To download models:

```bash
# Option 1: Use download script
./scripts/download_models.sh

# Option 2: Models will be automatically downloaded when running tests
python -m pytest tests/test_vgg16_formal.py
```

The models will be saved to the `models/` directory.

## 📚 Documentation

Full documentation is available at [abstractnn.readthedocs.io](https://abstractnn.readthedocs.io)

- **[Installation Guide](https://abstractnn.readthedocs.io/en/latest/installation.html)**
- **[Quick Start Tutorial](https://abstractnn.readthedocs.io/en/latest/quickstart.html)**
- **[API Reference](https://abstractnn.readthedocs.io/en/latest/api/modules.html)**
- **[Examples](https://abstractnn.readthedocs.io/en/latest/tutorials/index.html)**

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=abstractnn --cov-report=html

# Run specific test
pytest tests/test_affine_engine.py -v
```

## 🏗️ Architecture

abstractNN/
├── abstractnn/           # Main package
│   ├── __init__.py
│   ├── affine_engine.py  # Affine expression management
│   ├── bound_propagator.py  # Layer-by-layer propagation
│   ├── relaxer.py        # Non-linear relaxations
│   ├── onnx_parser.py    # ONNX model parsing
│   ├── partial_evaluator.py  # Partial network evaluation
│   ├── soundness_checker.py  # Soundness validation
│   └── cli.py            # Command-line interface
├── tests/                # Test suite
├── docs/                 # Sphinx documentation
├── examples/             # Usage examples
└── scripts/              # Utility scripts

##🔬 Research & Citations

If you use abstractNN in your research, please cite:

```cite
@software{abstractnn2025,
  title={abstractNN: Formal Verification of Neural Networks using Abstract Interpretation},
  author={Berthelot, Guillaume},
  year={2025},
  url={https://github.com/guillaume117/abstractNN},
  version={0.1.0}
}
```

## 🤝 Contributing
Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

```bash
# Setup development environment
git clone https://github.com/flyworthi/abstractNN.git
cd abstractNN
pip install -e .[dev]# Run testspytest tests/# Format codeblack abstractnn/ tests/isort abstractnn/ tests/# Type checkingmypy abstractnn/```## 📊 Comparison with Other Tools

Tool	Soundness	Scalability	Speed	Tightness
abstractNN	✅ Yes	⚠️ Medium	⚠️ Medium	⚠️ Good
ERAN	✅ Yes	✅ High	✅ Fast	⚠️ Good
Marabou	✅ Yes	❌ Low	❌ Slow	✅ Exact
α,β-CROWN	✅ Yes	✅ High	✅ Fast	✅ GoodabstractNN/
├── abstractnn/           # Main package
│   ├── __init__.py
│   ├── affine_engine.py  # Affine expression management
│   ├── bound_propagator.py  # Layer-by-layer propagation
│   ├── relaxer.py        # Non-linear relaxations
│   ├── onnx_parser.py    # ONNX model parsing
│   ├── partial_evaluator.py  # Partial network evaluation
│   ├── soundness_checker.py  # Soundness validation
│   └── cli.py            # Command-line interface
├── tests/                # Test suite
├── docs/                 # Sphinx documentation
├── examples/             # Usage examples
└── scripts/              # Utility scripts
