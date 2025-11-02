# Changelog

All notable changes to abstractNN will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Release Notes

### v0.1.2 - Initial Release

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
- Untighted relaxation for non linear function.


**Migration Guide:** N/A (first release)

**Contributors:**
- Guillaume Berthelot (@flyworthi)

**Acknowledgments:**
- Research inspired by abstract interpretation techniques
- Built with PyTorch, ONNX, and NumPy

---
