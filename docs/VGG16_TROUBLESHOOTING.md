# Guide de Dépannage - Tests VGG16

## Problèmes Courants

### 1. Erreur de Type : "Input type (double) and bias type (float)"

**Cause** : Incompatibilité entre les types numpy (float64/double) et PyTorch (float32).

**Solution** :
```python
# ❌ Incorrect
img = np.random.rand(3, 224, 224)  # float64 par défaut

# ✅ Correct
img = np.random.rand(3, 224, 224).astype(np.float32)

# ✅ Correct lors de la création de tenseur
input_tensor = torch.from_numpy(image.astype(np.float32))
```

### 2. Avertissement "pretrained parameter is deprecated"

**Cause** : API torchvision mise à jour.

**Solution** :
```python
# ❌ Ancienne API
model = models.vgg16(pretrained=True)

# ✅ Nouvelle API
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
```

### 3. Erreur ONNX Version Converter

**Cause** : Incompatibilité entre opset versions.

**Solution** :
```python
# Utiliser opset version 17 au lieu de 11
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=17  # ✅ Au lieu de 11
)
```

### 4. Erreur JSON : "Object of type bool_ is not JSON serializable"

**Cause** : Types NumPy (np.bool_, np.int64, np.float64) ne sont pas sérialisables en JSON.

**Solution** :
```python
# ❌ Incorrect
results = {
    'robust': original_class == noisy_class,  # Type numpy.bool_
    'class_id': predicted_class  # Type numpy.int64
}

# ✅ Correct
results = {
    'robust': bool(original_class == noisy_class),  # Conversion explicite
    'class_id': int(predicted_class),
    'probability': float(prob),
    'epsilon': float(epsilon)
}

# ✅ Alternative : Fonction utilitaire
def numpy_to_python(obj):
    """Convertir types numpy en types Python natifs"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
```

### 5. Mémoire Insuffisante

**Symptôme** : `MemoryError` ou système bloqué.

**Solutions** :

#### Option A : Réduire la résolution
```python
# Tester avec 112x112 au lieu de 224x224
image = create_test_image(shape=(112, 112))
```

#### Option B : Traiter par batch plus petits
```python
# Monte Carlo par petits lots
batch_size = 10
for i in range(0, num_samples, batch_size):
    batch = process_batch(images[i:i+batch_size])
```

#### Option C : Libérer la mémoire explicitement
```python
import gc
import torch

# Après chaque test lourd
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 6. Tests Lents

**Cause** : Grand nombre d'échantillons Monte Carlo.

**Solution** :
```python
# Réduire temporairement pour debugging
num_samples = 10  # Au lieu de 100

# Utiliser multiprocessing
from multiprocessing import Pool

def process_sample(args):
    image, noise_level = args
    return standard_inference(image + noise_level)

with Pool(4) as pool:
    results = pool.map(process_sample, samples)
```

## Optimisations

### 1. Cache des Poids

```python
# Télécharger une seule fois
if not os.path.exists('models/vgg16_weights.pth'):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), 'models/vgg16_weights.pth')
else:
    model = models.vgg16(weights=None)
    model.load_state_dict(torch.load('models/vgg16_weights.pth'))
```

### 2. Inférence GPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Lors de l'inférence
input_tensor = input_tensor.to(device)
output = model(input_tensor)
```

### 3. Batch Processing

```python
# Traiter plusieurs images en parallèle
images_batch = torch.stack([img1, img2, img3, img4])
outputs = model(images_batch)
```

### 4. Sérialisation JSON Robuste

```python
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour types NumPy"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Utilisation
with open('results.json', 'w') as f:
    json.dump(data, f, cls=NumpyEncoder, indent=2)
```

## Vérification de l'Environnement

### Script de Diagnostic

```python
import sys
import torch
import torchvision
import numpy as np

print("=== Diagnostic Environnement ===")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"NumPy: {np.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test de compatibilité types
img_np = np.random.rand(3, 224, 224).astype(np.float32)
tensor = torch.from_numpy(img_np)
print(f"\nType NumPy: {img_np.dtype}")
print(f"Type Tensor: {tensor.dtype}")
print("✅ Compatibilité OK" if tensor.dtype == torch.float32 else "❌ Problème de type")

# Test sérialisation JSON
test_data = {
    'bool_numpy': np.bool_(True),
    'int_numpy': np.int64(42),
    'float_numpy': np.float32(3.14),
    'bool_python': True,
    'int_python': 42,
    'float_python': 3.14
}

try:
    import json
    json_str = json.dumps(test_data)
    print("\n❌ Types NumPy sérialisables (comportement inattendu)")
except TypeError as e:
    print(f"\n⚠️  Types NumPy non sérialisables : {e}")
    print("   Solution : Convertir explicitement avec int(), float(), bool()")
```

## Commandes Utiles

```bash
# Vérifier les dépendances
pip list | grep -E 'torch|numpy|onnx'

# Nettoyer le cache
rm -rf __pycache__ tests/__pycache__ modules/__pycache__
rm -rf models/*.onnx  # Re-exporter le modèle

# Exécuter un seul test
python -m unittest tests.test_vgg16_formal.TestVGG16Formal.test_02_standard_inference -v

# Exécuter un test spécifique avec plus de détails
python -m unittest tests.test_vgg16_formal.TestVGG16Formal.test_07_robustness_metrics -v

# Profiler la mémoire
python -m memory_profiler tests/test_vgg16_formal.py

# Profiler le temps
python -m cProfile -o profile.stats tests/test_vgg16_formal.py
```

## Résolution de Problèmes Spécifiques

### ImportError pour les modules

```bash
# S'assurer que le PYTHONPATH est correct
export PYTHONPATH=/home/dopamine/AbstClaud:$PYTHONPATH
```

### Problèmes de permissions

```bash
# Donner les droits d'exécution au script
chmod +x scripts/run_vgg16_test.sh

# Créer les dossiers avec les bonnes permissions
mkdir -p models results data logs
chmod 755 models results data logs
```

### Vérifier la sérialisation JSON

```python
# Script de test rapide
import json
import numpy as np

def test_json_serialization():
    """Tester la sérialisation de différents types"""
    test_cases = {
        'numpy_bool': np.bool_(True),
        'numpy_int': np.int64(42),
        'numpy_float': np.float32(3.14),
        'python_bool': True,
        'python_int': 42,
        'python_float': 3.14
    }
    
    for key, value in test_cases.items():
        try:
            json.dumps({key: value})
            print(f"✅ {key} ({type(value).__name__}): OK")
        except TypeError:
            print(f"❌ {key} ({type(value).__name__}): FAIL")
            # Tester la conversion
            converted = type(value).__name__.replace('_', '').replace('numpy', '')
            if 'bool' in converted:
                print(f"   → Utiliser: bool({value})")
            elif 'int' in converted:
                print(f"   → Utiliser: int({value})")
            elif 'float' in converted:
                print(f"   → Utiliser: float({value})")

test_json_serialization()
```

## Support

Si le problème persiste :
1. Vérifier les versions des dépendances
2. Nettoyer tous les caches
3. Ré-exporter le modèle ONNX
4. Consulter les logs détaillés dans `logs/`
5. Vérifier les types des données avant sérialisation JSON

## Checklist de Débogage

- [ ] Types NumPy convertis en types Python natifs
- [ ] Images en float32 (pas float64)
- [ ] Modèle chargé avec nouvelle API weights
- [ ] ONNX exporté avec opset 17+
- [ ] Dossiers créés (models, results, logs)
- [ ] Permissions correctes sur les scripts
- [ ] PYTHONPATH configuré si nécessaire
- [ ] Mémoire suffisante (au moins 8GB recommandé)
