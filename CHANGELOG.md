# Changelog

All notable changes to abstractNN will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- GPU acceleration with CuPy
- Parallel bound propagation
- Expression simplification optimization
- Support for more ONNX operations
- Interactive visualization tools

## [0.1.0] - 2024-01-15

### Added
- Initial release of abstractNN
- Core affine expression engine
- Bound propagator for layer-by-layer verification
- Non-linear relaxation strategies (linear, interval, hybrid)
- ONNX model parser
- Partial network evaluator for large models
- Soundness checker with Monte Carlo validation
- Command-line interface (abstractnn-verify, abstractnn-info)
- Comprehensive test suite
- Sphinx documentation
- VGG16 verification examples

### Features
- Support for Conv2d, Linear, ReLU, MaxPool layers
- Mathematically sound bound computation
- Partial evaluation for memory efficiency
- ONNX model import
- Type hints throughout codebase
- PyPI package distribution

### Documentation
- Installation guide
- Quick start tutorial
- API reference
- User guides
- VGG16 case study
- Troubleshooting guide

### Tests
- Unit tests for all core modules
- Integration tests for VGG16
- Soundness verification tests
- 80%+ code coverage

## [0.0.1] - 2024-01-01

### Added
- Project initialization
- Basic project structure
- Initial module skeleton

---

## Release Notes

### v0.1.0 - Initial Release

This is the first public release of abstractNN, a formal verification library for neural networks using abstract interpretation and affine arithmetic.

**Highlights:**

✨ **Core Functionality**
- Sound verification of neural networks
- Affine arithmetic-based bound propagation
- Support for common layer types (Conv, Linear, ReLU, MaxPool)

📦 **Easy Installation**
- Available on PyPI: `pip install abstractNN`
- Optional extras for development, documentation, GPU, and visualization

🧪 **Well Tested**
- Comprehensive test suite
- VGG16 case study
- Monte Carlo soundness validation

📚 **Documentation**
- Full Sphinx documentation
- API reference
- Tutorials and examples

🛠️ **Developer Friendly**
- Type hints throughout
- Code formatted with Black
- Continuous integration

**Known Limitations:**
- Memory intensive for large networks (VGG16 full network)
- Some ONNX operations not yet supported
- No GPU acceleration yet

**Migration Guide:** N/A (first release)

**Contributors:**
- Guillaume Berthelot (@flyworthi)

**Acknowledgments:**
- Research inspired by abstract interpretation techniques
- Built with PyTorch, ONNX, and NumPy

---

[Unreleased]: https://github.com/flyworthi/abstractNN/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/flyworthi/abstractNN/releases/tag/v0.1.0
[0.0.1]: https://github.com/flyworthi/abstractNN/releases/tag/v0.0.1
