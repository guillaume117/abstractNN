# Model Files

This directory contains pre-trained neural network models used for testing and examples.

## Excluded from Git

Large model files (`.onnx`, `.pth`, `.pt`) are excluded from the Git repository due to GitHub's 100 MB file size limit.

## Downloading Models

### Automatic Download

Models will be automatically downloaded when you run the tests:

```bash
python -m pytest tests/test_vgg16_formal.py
```

### Manual Download

Use the provided script:

```bash
chmod +x ../scripts/download_models.sh
../scripts/download_models.sh
```

### Programmatic Download

```python
import torch
import torchvision.models as models

# Download VGG16
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'models/vgg16.onnx',
    export_params=True,
    opset_version=17,
    input_names=['input'],
    output_names=['output']
)
```

## Available Models

- **VGG16** (`vgg16.onnx`) - ~528 MB
  - ImageNet pre-trained
  - Used for integration tests
  - Auto-downloaded from PyTorch Hub

## Storage Options

If you need to store large models:

1. **Git LFS** (GitHub Large File Storage)
2. **External Storage** (Google Drive, Dropbox)
3. **Model Zoo** (Hugging Face, PyTorch Hub)

## Size Information

- `vgg16.onnx` + `vgg16.onnx.data`: ~528 MB
- Total models directory: ~550 MB (when populated)
