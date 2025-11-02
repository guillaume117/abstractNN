"""
Script de test de soundness pour le moteur d'évaluation formelle
"""

import os
import sys
from abstractnn.affine_eval import evaluate_model
from abstractnn.soundness_checker import SoundnessChecker, monte_carlo_robustness_test


def test_soundness_single_image(model_path: str, 
                                image_path: str,
                                noise_level: float = 0.00005,
                                num_samples: int = 1000):
    """
    Test de soundness sur une seule image
    """
    
    print("="*70)
    print("TEST DE SOUNDNESS - ÉVALUATION FORMELLE")
    print("="*70)
    print(f"Modèle: {model_path}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Niveau de bruit: ±{noise_level}")
    print("="*70)
    
    # 1. Évaluation formelle
    print("\n[1/3] Évaluation formelle...")
    formal_results = evaluate_model(
        model_path=model_path,
        input_image=image_path,
        noise_level=noise_level,
        activation_relaxation='linear'
    )
    
    bounds = formal_results['bounds_per_class']
    
    # Affiche les bornes formelles
    print("\nBornes formelles calculées:")
    for class_name, (lower, upper) in sorted(bounds.items()):
        print(f"  {class_name}: [{lower:.4f}, {upper:.4f}]")
    
    # 2. Vérification de soundness
    print("\n[2/3] Vérification de soundness...")
    checker = SoundnessChecker(model_path)
    soundness_results = checker.check_soundness(
        image_path=image_path,
        bounds=bounds,
        noise_level=noise_level,
        num_samples=num_samples
    )
    
    checker.print_soundness_report(soundness_results)
    
    # 3. Test Monte Carlo
    print("\n[3/3] Test Monte Carlo de robustesse...")
    mc_results = monte_carlo_robustness_test(
        model_path=model_path,
        image_path=image_path,
        noise_level=noise_level,
        num_samples=num_samples
    )
    
    # Comparaison
    print("\n" + "="*70)
    print("COMPARAISON FORMEL vs EMPIRIQUE")
    print("="*70)
    
    formal_robust = formal_results['robust_class'] is not None
    empirical_robust = mc_results['empirical_robustness'] > 0.95
    
    print(f"\nRobustesse formelle certifiée: {'OUI' if formal_robust else 'NON'}")
    print(f"Robustesse empirique (>95%): {'OUI' if empirical_robust else 'NON'}")
    
    if formal_robust and empirical_robust:
        print("\n✓ Cohérence: Les deux méthodes confirment la robustesse")
    elif formal_robust and not empirical_robust:
        print("\n⚠️  Incohérence: Formel dit robuste mais empirique non")
        print("   (Cela peut indiquer un problème dans l'implémentation)")
    elif not formal_robust and empirical_robust:
        print("\n✓ Conservatif: Formel plus prudent que l'empirique")
        print("   (Normal: l'approche formelle est conservative)")
    else:
        print("\n✓ Cohérence: Les deux méthodes confirment la non-robustesse")
    
    return {
        'formal': formal_results,
        'soundness': soundness_results,
        'monte_carlo': mc_results
    }


def test_soundness_multiple_noise_levels(model_path: str,
                                        image_path: str,
                                        noise_levels: list = None):
    """
    Test de soundness pour plusieurs niveaux de bruit
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.03, 0.05, 0.1]
    
    print("="*70)
    print("TEST DE SOUNDNESS - MULTIPLES NIVEAUX DE BRUIT")
    print("="*70)
    
    all_results = []
    
    for noise in noise_levels:
        print(f"\n{'='*70}")
        print(f"NIVEAU DE BRUIT: ε = {noise:.3f}")
        print(f"{'='*70}")
        
        results = test_soundness_single_image(
            model_path=model_path,
            image_path=image_path,
            noise_level=noise,
            num_samples=500  # Moins d'échantillons pour aller plus vite
        )
        
        all_results.append({
            'noise_level': noise,
            'is_sound': results['soundness']['is_sound'],
            'violation_rate': results['soundness']['violation_rate'],
            'empirical_robustness': results['monte_carlo']['empirical_robustness']
        })
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ - SOUNDNESS PAR NIVEAU DE BRUIT")
    print("="*70)
    print(f"\n{'Bruit (ε)':<12} | {'Sound':<8} | {'Violations':<12} | {'Robustesse MC':<15}")
    print("-"*70)
    
    for r in all_results:
        sound_mark = '✓' if r['is_sound'] else '✗'
        print(f"{r['noise_level']:<12.3f} | {sound_mark:<8} | "
              f"{r['violation_rate']*100:<11.4f}% | {r['empirical_robustness']*100:<14.2f}%")
    
    return all_results


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
    images = [f for f in os.listdir(image_dir) 
              if f.startswith('fmnist_sample_') and f.endswith('.png')]
    
    if not images:
        print("Erreur: Aucune image de test trouvée.")
        print("Veuillez d'abord exécuter: python create_fmnist_model.py")
        return
    
    test_image = os.path.join(image_dir, images[0])
    
    # Menu
    print("\n" + "="*70)
    print("TEST DE SOUNDNESS DU MOTEUR D'ÉVALUATION FORMELLE")
    print("="*70)
    print("\nOptions:")
    print("  1. Test simple (1 niveau de bruit, 1000 échantillons)")
    print("  2. Test approfondi (1 niveau de bruit, 10000 échantillons)")
    print("  3. Test multi-niveaux (4 niveaux de bruit, 500 échantillons chacun)")
    
    choice = input("\nChoisissez une option (1-3) [1]: ").strip() or "1"
    
    if choice == "1":
        test_soundness_single_image(model_path, test_image, noise_level=0.05, num_samples=1000)
    elif choice == "2":
        test_soundness_single_image(model_path, test_image, noise_level=0.00005, num_samples=10000)
    elif choice == "3":
        test_soundness_multiple_noise_levels(model_path, test_image)
    else:
        print("Option invalide.")
    
    print("\n" + "="*70)
    print("Tests terminés !")
    print("="*70)


if __name__ == '__main__':
    main()
