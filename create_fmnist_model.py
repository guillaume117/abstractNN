"""
Script pour créer et entraîner un CNN simple pour Fashion-MNIST
et l'exporter au format ONNX
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image


class FashionMNISTCNN(nn.Module):
    """CNN simple pour Fashion-MNIST"""
    
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        
        # Couche conv1: 1 -> 16 channels, kernel 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Couche conv2: 16 -> 32 channels, kernel 3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Couches fully connected
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Fully Connected
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x


def train_model(epochs=3, batch_size=64):
    """Entraîne le modèle sur Fashion-MNIST"""
    
    # Prépare les données
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Crée le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FashionMNISTCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Entraînement sur {device}")
    print(f"Architecture du modèle:")
    print(model)
    print(f"\nNombre de paramètres: {sum(p.numel() for p in model.parameters())}")
    
    # Entraînement
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)] '
                      f'Loss: {loss.item():.6f}')
        
        train_acc = 100. * correct / total
        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%')
        
        # Test
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * correct / total
        print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.2f}%\n')
    
    return model


def export_to_onnx(model, output_path='examples/fmnist_cnn.onnx'):
    """Exporte le modèle au format ONNX"""
    
    # Force le modèle et l'entrée sur CPU pour l'export
    model = model.cpu()
    model.eval()
    
    # Entrée exemple (batch_size=1, channels=1, height=28, width=28)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Export ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Modèle exporté vers: {output_path}")


def save_sample_images(num_samples=5, output_dir='examples'):
    """Sauvegarde des images d'exemple de Fashion-MNIST"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Charge le dataset de test
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Sauvegarde des échantillons
    for i in range(num_samples):
        img, label = test_dataset[i]
        
        # Convertit en PIL Image
        img_array = (img.squeeze().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array, mode='L')
        
        # Sauvegarde
        filename = f"{output_dir}/fmnist_sample_{i}_{class_names[label].replace('/', '_')}.png"
        pil_img.save(filename)
        print(f"Sauvegardé: {filename} (classe: {class_names[label]})")


def main():
    """Point d'entrée principal"""
    
    import os
    os.makedirs('examples', exist_ok=True)
    
    print("=" * 60)
    print("Création d'un modèle CNN pour Fashion-MNIST")
    print("=" * 60)
    
    # Entraîne le modèle
    print("\n1. Entraînement du modèle...")
    model = train_model(epochs=3, batch_size=64)
    
    # Exporte en ONNX (le modèle sera mis sur CPU automatiquement)
    print("\n2. Export au format ONNX...")
    export_to_onnx(model)
    
    # Sauvegarde des images d'exemple
    print("\n3. Sauvegarde d'images d'exemple...")
    save_sample_images(num_samples=10)
    
    print("\n" + "=" * 60)
    print("Terminé !")
    print("Fichiers créés:")
    print("  - examples/fmnist_cnn.onnx (modèle)")
    print("  - examples/fmnist_sample_*.png (images de test)")
    print("\nVous pouvez maintenant tester avec:")
    print("  python affine_eval.py \\")
    print("    --model examples/fmnist_cnn.onnx \\")
    print("    --image examples/fmnist_sample_0_Ankle_boot.png \\")
    print("    --noise 0.05 \\")
    print("    --output results.json")
    print("=" * 60)


if __name__ == '__main__':
    main()
