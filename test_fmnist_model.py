"""
Script de test pour le modèle Fashion-MNIST avec le moteur d'évaluation formelle
"""

import os
import json
import numpy as np
from abstractnn.affine_eval import evaluate_model
from PIL import Image


def test_single_image(model_path, image_path, noise_level=0.05):
    """Teste une seule image"""
    
    print(f"\n{'='*60}")
    print(f"Test: {os.path.basename(image_path)}")
    print(f"Niveau de bruit: ±{noise_level}")
    print(f"{'='*60}\n")
    
    # Évalue avec le moteur formel et rapport détaillé
    results = evaluate_model(
        model_path=model_path,
        input_image=image_path,
        noise_level=noise_level,
        activation_relaxation='linear',
        enable_detailed_report=True  # Active le rapport détaillé
    )
    
    # Affiche les résultats
    print("\n" + "-" * 60)
    print("RÉSULTATS DE L'ÉVALUATION FORMELLE")
    print("-" * 60)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    bounds = results['bounds_per_class']
    
    # Affiche les bornes par classe
    print("\nBornes certifiées par classe:")
    for class_name, (lower, upper) in sorted(bounds.items()):
        class_idx = int(class_name.split('_')[1])
        name = class_names[class_idx]
        print(f"  {name:15s} [{lower:8.4f}, {upper:8.4f}]  (largeur: {upper-lower:.4f})")
    
    # Trouve la classe prédite (borne inférieure maximale)
    predicted_class = max(bounds.items(), key=lambda x: x[1][0])[0]
    pred_idx = int(predicted_class.split('_')[1])
    
    print(f"\nClasse prédite: {class_names[pred_idx]}")
    print(f"Classe robuste: {results['robust_class'] if results['robust_class'] else 'Aucune (non certifiée)'}")
    print(f"Temps d'exécution: {results['execution_time']:.2f}s")
    
    return results


def test_robustness_sweep(model_path, image_path, noise_levels=[0.01, 0.03, 0.05, 0.1, 0.15]):
    """Teste la robustesse pour différents niveaux de bruit"""
    
    print(f"\n{'='*60}")
    print("ANALYSE DE ROBUSTESSE (SWEEP)")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"{'='*60}\n")
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    results_sweep = []
    
    for noise in noise_levels:
        print(f"\nTest avec bruit ε = {noise:.3f}")
        print("-" * 40)
        
        try:
            results = evaluate_model(
                model_path=model_path,
                input_image=image_path,
                noise_level=noise,
                activation_relaxation='linear'
            )
            
            bounds = results['bounds_per_class']
            predicted_class = max(bounds.items(), key=lambda x: x[1][0])[0]
            pred_idx = int(predicted_class.split('_')[1])
            
            # Calcule la marge de robustesse
            pred_lower = bounds[predicted_class][0]
            other_uppers = [upper for cls, (_, upper) in bounds.items() if cls != predicted_class]
            max_other_upper = max(other_uppers)
            margin = pred_lower - max_other_upper
            
            is_robust = results['robust_class'] is not None
            
            result_entry = {
                'noise': noise,
                'predicted_class': class_names[pred_idx],
                'is_certified_robust': is_robust,
                'robustness_margin': margin,
                'execution_time': results['execution_time']
            }
            
            results_sweep.append(result_entry)
            
            print(f"  Classe prédite: {class_names[pred_idx]}")
            print(f"  Certifiée robuste: {'OUI' if is_robust else 'NON'}")
            print(f"  Marge: {margin:.4f}")
            print(f"  Temps: {results['execution_time']:.2f}s")
            
        except Exception as e:
            print(f"  Erreur: {e}")
    
    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    print(f"\n{'Bruit (ε)':>12} | {'Classe prédite':>15} | {'Robuste':>8} | {'Marge':>10} | {'Temps (s)':>10}")
    print("-" * 70)
    
    for r in results_sweep:
        print(f"{r['noise']:>12.3f} | {r['predicted_class']:>15} | "
              f"{'✓' if r['is_certified_robust'] else '✗':>8} | "
              f"{r['robustness_margin']:>10.4f} | {r['execution_time']:>10.2f}")
    
    return results_sweep


def compare_with_onnx_inference(model_path, image_path):
    """Compare avec l'inférence ONNX standard"""
    
    import onnxruntime as ort
    
    print(f"\n{'='*60}")
    print("COMPARAISON AVEC INFÉRENCE STANDARD")
    print(f"{'='*60}\n")
    
    # Charge l'image
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 1, 28, 28)
    
    # Inférence ONNX
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})[0]
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print("Scores de sortie (inférence standard):")
    for i, score in enumerate(output[0]):
        print(f"  {class_names[i]:15s}: {score:8.4f}")
    
    predicted_idx = np.argmax(output[0])
    print(f"\nClasse prédite: {class_names[predicted_idx]} (score: {output[0][predicted_idx]:.4f})")


def main():
    """Point d'entrée principal"""
    
    model_path = 'examples/fmnist_cnn.onnx'
    
    # Vérifie que le modèle existe
    if not os.path.exists(model_path):
        print("Erreur: Le modèle n'existe pas.")
        print("Veuillez d'abord exécuter: python create_fmnist_model.py")
        return
    
    # Liste les images disponibles
    image_dir = 'examples'
    images = [f for f in os.listdir(image_dir) if f.startswith('fmnist_sample_') and f.endswith('.png')]
    
    if not images:
        print("Erreur: Aucune image de test trouvée.")
        print("Veuillez d'abord exécuter: python create_fmnist_model.py")
        return
    
    # Test sur la première image
    test_image = os.path.join(image_dir, images[0])
    
    print("="*60)
    print("TEST DU MOTEUR D'ÉVALUATION FORMELLE")
    print("Modèle: Fashion-MNIST CNN")
    print("="*60)
    
    # 1. Comparaison avec inférence standard
    compare_with_onnx_inference(model_path, test_image)
    
    # 2. Test avec un niveau de bruit fixe
    test_single_image(model_path, test_image, noise_level=0.05)
    
    # 3. Analyse de robustesse
    test_robustness_sweep(model_path, test_image)
    
    print("\n" + "="*60)
    print("Tests terminés !")
    print("="*60)


if __name__ == '__main__':
    main()
