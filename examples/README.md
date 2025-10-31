# Examples

This directory contains example models and images for testing abstractNN.

## Fashion-MNIST CNN Example

The default example uses a Fashion-MNIST CNN classifier.

### Files

- `fmnist_cnn.onnx` - Pre-trained Fashion-MNIST CNN model
- `fmnist_sample_0_Ankle_boot.png` - Sample image (Ankle boot, class 0)

### Quick Start

```bash
# Run with default parameters
abstractnn-eval

# This is equivalent to:
abstractnn-eval \
    --model examples/fmnist_cnn.onnx \
    --image examples/fmnist_sample_0_Ankle_boot.png \
    --noise 0.05 \
    --output results.json
```

### Creating Your Own Examples

To add your own model:

1. Export your model to ONNX format
2. Place the `.onnx` file in this directory
3. Add a sample image
4. Run evaluation:

```bash
abstractnn-eval \
    --model examples/your_model.onnx \
    --image examples/your_image.png \
    --noise 0.1
```

## Fashion-MNIST Classes

- 0: Ankle boot
- 1: T-shirt/top
- 2: Trouser
- 3: Pullover
- 4: Dress
- 5: Coat
- 6: Sandal
- 7: Shirt
- 8: Sneaker
- 9: Bag

## Generating More Examples

```python
import torch
import torchvision
from PIL import Image

# Load Fashion-MNIST dataset
dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True
)

# Save a sample
img, label = dataset[0]
img.save(f'fmnist_sample_{label}_class.png')
```
