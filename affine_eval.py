"""
Point d'entrée principal du moteur d'évaluation formelle
"""

import argparse
import json
import time
import numpy as np
from PIL import Image
from pathlib import Path

from modules.onnx_parser import ONNXParser
from modules.affine_engine import AffineExpressionEngine
from modules.bound_propagator import BoundPropagator
from modules.result_aggregator import ResultAggregator


def load_image(image_path: str, target_shape: tuple = None) -> np.ndarray:
    """Charge et prétraite une image"""
    img = Image.open(image_path)
    
    # Détermine le nombre de canaux attendus
    expected_channels = 3  # Par défaut RGB
    if target_shape and len(target_shape) >= 2:
        if isinstance(target_shape[1], int):
            expected_channels = target_shape[1]
    
    # Convertit dans le bon format
    if expected_channels == 1:
        img = img.convert('L')  # Niveaux de gris
    else:
        img = img.convert('RGB')
    
    # Redimensionne si nécessaire
    if target_shape and len(target_shape) >= 4:
        target_h, target_w = target_shape[2], target_shape[3]
        img = img.resize((target_w, target_h))
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Ajoute la dimension channel si nécessaire (pour images en niveaux de gris)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=0)
    else:
        # Transpose pour correspondre à (channels, height, width)
        img_array = np.transpose(img_array, (2, 0, 1))
    
    # Ajoute la dimension batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def evaluate_model(model_path: str,
                   input_image: str,
                   noise_level: float,
                   activation_relaxation: str = 'linear',
                   device: str = 'cpu',
                   enable_detailed_report: bool = False) -> dict:
    """
    Évalue formellement un modèle ONNX avec perturbations bornées
    
    Args:
        model_path: Chemin vers le modèle ONNX
        input_image: Chemin vers l'image d'entrée
        noise_level: Niveau de bruit (epsilon)
        activation_relaxation: Type de relaxation ('linear', 'quadratic')
        device: Device à utiliser ('cpu' ou 'gpu')
        enable_detailed_report: Génère un rapport détaillé sur la propagation
    
    Returns:
        Dictionnaire contenant les bornes par classe et les métriques
    """
    start_time = time.time()
    
    # 1. Parse le modèle ONNX
    print(f"Chargement du modèle: {model_path}")
    parser = ONNXParser(model_path)
    layers = parser.parse()
    print(f"  {len(layers)} couches détectées")
    
    # 2. Charge l'image
    print(f"Chargement de l'image: {input_image}")
    input_shape = parser.get_input_shape()
    image_data = load_image(input_image, input_shape)
    print(f"  Forme de l'image: {image_data.shape}")
    
    # 3. Crée les expressions affines initiales
    print(f"Création des expressions affines (bruit: ±{noise_level})")
    engine = AffineExpressionEngine()
    initial_expressions = engine.create_input_expressions(image_data, noise_level)
    print(f"  {len(initial_expressions)} expressions initiales")
    
    # 4. Propage à travers le réseau
    print("Propagation à travers le réseau...")
    propagator = BoundPropagator(
        relaxation_type=activation_relaxation,
        enable_reporting=enable_detailed_report
    )
    output_expressions = propagator.propagate(
        initial_expressions, 
        layers,
        input_shape=image_data.shape
    )
    print(f"  {len(output_expressions)} sorties")
    
    # 5. Affiche le rapport détaillé si demandé
    if enable_detailed_report and propagator.reporter:
        propagator.reporter.print_detailed_report()
        propagator.reporter.print_complexity_analysis()
    
    # 6. Agrège les résultats
    print("Calcul des bornes finales...")
    aggregator = ResultAggregator()
    bounds_per_class = aggregator.compute_class_bounds(output_expressions)
    robust_class = aggregator.find_robust_class(bounds_per_class)
    
    execution_time = time.time() - start_time
    
    # 7. Prépare les résultats
    results = {
        'bounds_per_class': bounds_per_class,
        'robust_class': robust_class,
        'intermediate_bounds': propagator.intermediate_bounds,
        'execution_time': execution_time,
        'noise_level': noise_level,
        'activation_relaxation': activation_relaxation
    }
    
    # Ajoute le rapport si généré
    if enable_detailed_report and propagator.reporter:
        results['detailed_report'] = propagator.reporter.summary
    
    print(f"\nÉvaluation terminée en {execution_time:.2f}s")
    print(f"Classe robuste: {robust_class if robust_class else 'Aucune'}")
    
    return results


def convert_to_json_serializable(obj):
    """Convertit les types numpy en types Python natifs pour la sérialisation JSON"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(
        description='Moteur d\'évaluation formelle par expression affine'
    )
    parser.add_argument('--model', required=True, help='Chemin vers le modèle ONNX')
    parser.add_argument('--image', required=True, help='Chemin vers l\'image d\'entrée')
    parser.add_argument('--noise', type=float, required=True, 
                       help='Niveau de bruit (epsilon)')
    parser.add_argument('--activation-relaxation', default='linear',
                       choices=['linear', 'quadratic'],
                       help='Type de relaxation des activations')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu'],
                       help='Device à utiliser')
    parser.add_argument('--output', default='results.json',
                       help='Fichier de sortie pour les résultats')
    parser.add_argument('--detailed-report', action='store_true',
                       help='Génère un rapport détaillé sur la propagation')
    parser.add_argument('--export-report-csv', type=str,
                       help='Export le rapport en CSV')
    
    args = parser.parse_args()
    
    # Évalue le modèle
    results = evaluate_model(
        model_path=args.model,
        input_image=args.image,
        noise_level=args.noise,
        activation_relaxation=args.activation_relaxation,
        device=args.device,
        enable_detailed_report=args.detailed_report
    )
    
    # Sauvegarde les résultats
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        # Convertit tous les types en types sérialisables JSON
        json_results = convert_to_json_serializable(results)
        json.dump(json_results, f, indent=2)
    
    print(f"\nRésultats sauvegardés dans: {output_path}")
    
    # Export CSV si demandé
    if args.export_report_csv and 'detailed_report' in results:
        # Note: le reporter n'est pas sérialisable directement
        # Il faudrait le récupérer depuis le propagator
        print(f"Export CSV: utilisez --detailed-report pour activer")


if __name__ == '__main__':
    main()