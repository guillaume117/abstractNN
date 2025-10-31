"""
Vérification statistique de la soundness des bornes calculées
"""

import numpy as np
import onnxruntime as ort
from typing import Dict, Tuple, List
from PIL import Image


class SoundnessChecker:
    """Vérifie statistiquement que les bornes calculées sont correctes (sound)"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.violations = []
    
    def check_soundness(self,
                       image_path: str,
                       bounds: Dict[str, Tuple[float, float]],
                       noise_level: float,
                       num_samples: int = 1000) -> Dict:
        """
        Vérifie la soundness en échantillonnant aléatoirement dans la région bruitée
        
        Args:
            image_path: Chemin vers l'image d'entrée
            bounds: Bornes calculées par le moteur formel {class: (lower, upper)}
            noise_level: Niveau de bruit appliqué
            num_samples: Nombre d'échantillons à tester
        
        Returns:
            Dictionnaire contenant les statistiques de soundness
        """
        # Charge l'image de base
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        base_image = np.array(img, dtype=np.float32) / 255.0
        base_image = base_image.reshape(1, 1, 28, 28)
        
        # Statistiques
        violations_per_class = {cls: 0 for cls in bounds.keys()}
        samples_per_class = {cls: [] for cls in bounds.keys()}
        
        print(f"\nVérification de soundness avec {num_samples} échantillons...")
        
        # Échantillonne et teste
        for i in range(num_samples):
            # Génère une perturbation aléatoire (force float32)
            noise = np.random.uniform(-noise_level, noise_level, base_image.shape).astype(np.float32)
            perturbed_image = np.clip(base_image + noise, 0.0, 1.0).astype(np.float32)
            
            # Inférence
            output = self.session.run(None, {self.input_name: perturbed_image})[0][0]
            
            # Vérifie les bornes pour chaque classe
            for class_idx, (class_name, (lower_bound, upper_bound)) in enumerate(bounds.items()):
                actual_value = output[class_idx]
                samples_per_class[class_name].append(actual_value)
                
                # Vérifie si la valeur est dans les bornes (avec tolérance numérique)
                tolerance = 1e-6
                if actual_value < lower_bound - tolerance or actual_value > upper_bound + tolerance:
                    violations_per_class[class_name] += 1
                    self.violations.append({
                        'sample': i,
                        'class': class_name,
                        'actual': actual_value,
                        'bounds': (lower_bound, upper_bound),
                        'violation_type': 'lower' if actual_value < lower_bound else 'upper'
                    })
            
            if (i + 1) % 200 == 0:
                print(f"  Progression: {i + 1}/{num_samples} échantillons testés")
        
        # Calcule les statistiques
        results = {
            'num_samples': num_samples,
            'total_violations': sum(violations_per_class.values()),
            'violation_rate': sum(violations_per_class.values()) / (num_samples * len(bounds)),
            'violations_per_class': violations_per_class,
            'is_sound': sum(violations_per_class.values()) == 0,
            'statistics_per_class': {}
        }
        
        # Statistiques détaillées par classe
        for class_name, samples in samples_per_class.items():
            lower_bound, upper_bound = bounds[class_name]
            samples_array = np.array(samples)
            
            results['statistics_per_class'][class_name] = {
                'empirical_min': float(np.min(samples_array)),
                'empirical_max': float(np.max(samples_array)),
                'empirical_mean': float(np.mean(samples_array)),
                'empirical_std': float(np.std(samples_array)),
                'formal_lower': lower_bound,
                'formal_upper': upper_bound,
                'lower_margin': float(np.min(samples_array) - lower_bound),
                'upper_margin': float(upper_bound - np.max(samples_array)),
                'violations': violations_per_class[class_name]
            }
        
        return results
    
    def print_soundness_report(self, results: Dict):
        """Affiche un rapport de soundness lisible"""
        
        print("\n" + "="*70)
        print("RAPPORT DE SOUNDNESS")
        print("="*70)
        
        print(f"\nNombre d'échantillons testés: {results['num_samples']}")
        print(f"Violations totales: {results['total_violations']}")
        print(f"Taux de violation: {results['violation_rate']*100:.4f}%")
        print(f"Sound (correct): {'✓ OUI' if results['is_sound'] else '✗ NON'}")
        
        if results['is_sound']:
            print("\n🎉 Les bornes formelles sont SOUND (correctes) !")
            print("   Toutes les valeurs observées sont dans les bornes calculées.")
        else:
            print("\n⚠️  ATTENTION: Violations détectées !")
            print("   Certaines valeurs observées sont hors des bornes calculées.")
        
        print("\n" + "-"*70)
        print("STATISTIQUES PAR CLASSE")
        print("-"*70)
        print(f"\n{'Classe':<15} | {'Min obs':<10} | {'Max obs':<10} | {'Borne inf':<10} | {'Borne sup':<10} | {'Violations'}")
        print("-"*90)
        
        for class_name, stats in results['statistics_per_class'].items():
            class_idx = int(class_name.split('_')[1])
            violations = stats['violations']
            violation_mark = '✗' if violations > 0 else '✓'
            
            print(f"{class_name:<15} | {stats['empirical_min']:<10.4f} | "
                  f"{stats['empirical_max']:<10.4f} | {stats['formal_lower']:<10.4f} | "
                  f"{stats['formal_upper']:<10.4f} | {violations:>5} {violation_mark}")
        
        # Analyse des marges
        print("\n" + "-"*70)
        print("MARGES DE SÉCURITÉ")
        print("-"*70)
        print(f"\n{'Classe':<15} | {'Marge inf':<15} | {'Marge sup':<15} | {'Statut'}")
        print("-"*70)
        
        for class_name, stats in results['statistics_per_class'].items():
            lower_margin = stats['lower_margin']
            upper_margin = stats['upper_margin']
            
            if lower_margin < 0 or upper_margin < 0:
                status = "⚠️  VIOLATION"
            elif lower_margin < 1e-3 or upper_margin < 1e-3:
                status = "⚠️  Marge faible"
            else:
                status = "✓ OK"
            
            print(f"{class_name:<15} | {lower_margin:<15.6f} | {upper_margin:<15.6f} | {status}")
        
        # Affiche les violations si présentes
        if not results['is_sound'] and self.violations:
            print("\n" + "-"*70)
            print("DÉTAILS DES VIOLATIONS (premières 10)")
            print("-"*70)
            
            for violation in self.violations[:10]:
                lower, upper = violation['bounds']
                print(f"\nÉchantillon #{violation['sample']} - Classe {violation['class']}")
                print(f"  Valeur observée: {violation['actual']:.6f}")
                print(f"  Bornes formelles: [{lower:.6f}, {upper:.6f}]")
                print(f"  Type de violation: {violation['violation_type']}")
        
        print("\n" + "="*70)


def monte_carlo_robustness_test(model_path: str,
                                image_path: str,
                                noise_level: float,
                                num_samples: int = 10000) -> Dict:
    """
    Test Monte Carlo de robustesse empirique
    
    Compare avec l'évaluation formelle pour valider la conservativité
    
    Args:
        model_path: Chemin vers le modèle ONNX
        image_path: Chemin vers l'image
        noise_level: Niveau de bruit
        num_samples: Nombre d'échantillons Monte Carlo
    
    Returns:
        Statistiques de robustesse empirique
    """
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Charge l'image
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    base_image = np.array(img, dtype=np.float32) / 255.0
    base_image = base_image.reshape(1, 1, 28, 28)
    
    # Prédiction sur l'image originale
    original_output = session.run(None, {input_name: base_image})[0][0]
    original_class = np.argmax(original_output)
    
    # Échantillonnage Monte Carlo
    class_changes = 0
    predictions = []
    
    print(f"\nTest Monte Carlo avec {num_samples} échantillons...")
    
    for i in range(num_samples):
        # Force float32 pour éviter les erreurs de type
        noise = np.random.uniform(-noise_level, noise_level, base_image.shape).astype(np.float32)
        perturbed_image = np.clip(base_image + noise, 0.0, 1.0).astype(np.float32)
        
        output = session.run(None, {input_name: perturbed_image})[0][0]
        predicted_class = np.argmax(output)
        predictions.append(predicted_class)
        
        if predicted_class != original_class:
            class_changes += 1
        
        if (i + 1) % 2000 == 0:
            print(f"  Progression: {i + 1}/{num_samples}")
    
    # Statistiques
    predictions = np.array(predictions)
    empirical_robustness = 1.0 - (class_changes / num_samples)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    results = {
        'num_samples': num_samples,
        'original_class': int(original_class),
        'original_class_name': class_names[original_class],
        'class_changes': class_changes,
        'empirical_robustness': empirical_robustness,
        'prediction_distribution': {
            i: int(np.sum(predictions == i)) for i in range(10)
        }
    }
    
    print(f"\n{'='*60}")
    print("RÉSULTATS MONTE CARLO")
    print(f"{'='*60}")
    print(f"Classe originale: {class_names[original_class]}")
    print(f"Changements de classe: {class_changes}/{num_samples}")
    print(f"Robustesse empirique: {empirical_robustness*100:.2f}%")
    print(f"\nDistribution des prédictions:")
    for i, count in results['prediction_distribution'].items():
        if count > 0:
            print(f"  {class_names[i]}: {count} ({count/num_samples*100:.2f}%)")
    
    return results
