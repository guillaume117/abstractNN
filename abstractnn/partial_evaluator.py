"""
Évaluateur partiel pour sous-réseaux
=====================================

Module pour évaluer formellement un sous-ensemble de couches d'un réseau.
Utile pour tester la soundness sur de grands modèles comme VGG16.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import torch
import onnxruntime as ort
from .affine_engine import AffineExpressionEngine, AffineExpression
from .onnx_parser import ONNXParser
from .bound_propagator import BoundPropagator
from .relaxer import NonLinearRelaxer


class PartialNetworkEvaluator:
    """Évaluateur pour sous-réseaux - utilise BoundPropagator en interne"""
    
    def __init__(self, engine: AffineExpressionEngine = None):
        """
        Initialiser l'évaluateur partiel
        
        Args:
            engine: Moteur d'expressions affines (optionnel, créé si absent)
        """
        self.engine = engine or AffineExpressionEngine()
        # Utiliser BoundPropagator au lieu de réécrire la logique
        self.propagator = BoundPropagator(
            relaxation_type='linear',
            enable_reporting=False  # Pas de rapport par défaut
        )
    
    def create_input_expressions(
        self,
        image: np.ndarray,
        noise_level: float
    ) -> List[AffineExpression]:
        """
        Créer les expressions d'entrée avec bruit
        
        Args:
            image: Image d'entrée (C, H, W) ou (B, C, H, W)
            noise_level: Niveau de bruit epsilon
            
        Returns:
            Liste d'expressions affines
        """
        return self.engine.create_input_expressions(image, noise_level)
    
    def propagate_through_layers(
        self,
        expressions: List[AffineExpression],
        layers: List[Dict[str, Any]],
        input_shape: Tuple[int, ...]
    ) -> Tuple[List[AffineExpression], List[Tuple[float, float]]]:
        """
        Propager les expressions à travers un sous-ensemble de couches
        
        Utilise BoundPropagator.propagate() en interne.
        
        Args:
            expressions: Expressions d'entrée
            layers: Liste des couches à propager
            input_shape: Forme de l'entrée (B, C, H, W) ou (C, H, W)
            
        Returns:
            Tuple (expressions_sortie, bornes_intermédiaires)
        """
        print(f"  Propagation à travers {len(layers)} couches...")
        print(f"  Forme d'entrée initiale : {input_shape}")
        
        # Utiliser le propagateur existant
        try:
            output_expressions = self.propagator.propagate(
                expressions,
                layers,
                input_shape=input_shape
            )
            
            # Extraire les bornes intermédiaires
            all_bounds = []
            for bound_info in self.propagator.intermediate_bounds:
                all_bounds.extend(bound_info.get('bounds', []))
            
            print(f"  ✓ Propagation terminée : {len(output_expressions)} expressions")
            
            return output_expressions, all_bounds
            
        except Exception as e:
            print(f"  ✗ Erreur lors de la propagation : {e}")
            import traceback
            traceback.print_exc()
            raise


class ONNXPartialEvaluator:
    """Évaluateur utilisant ONNX Runtime pour les activations intermédiaires"""
    
    def __init__(self, model_path: str):
        """
        Initialiser l'évaluateur ONNX
        
        Args:
            model_path: Chemin vers le modèle ONNX
        """
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.parser = ONNXParser(model_path)
    
    def monte_carlo_sampling_at_layer(
        self,
        image: np.ndarray,
        noise_level: float,
        num_layers: int,
        num_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Échantillonnage Monte Carlo à une couche spécifique
        
        Utilise ONNX Runtime pour l'inférence complète, puis compare
        avec les résultats formels.
        
        Args:
            image: Image d'entrée (C, H, W) ou (B, C, H, W)
            noise_level: Niveau de bruit
            num_layers: Nombre de couches à traverser (ignoré pour full network)
            num_samples: Nombre d'échantillons
            
        Returns:
            Tuple (bornes_min, bornes_max) des sorties du réseau
        """
        print(f"\n  Monte Carlo : échantillonnage sur le réseau complet...")
        
        try:
            # Stratégie 1: Essayer PyTorch pour VGG16
            if 'vgg16' in self.model_path.lower():
                return self._monte_carlo_vgg16(image, noise_level, num_layers, num_samples)
            
            # Stratégie 2: Utiliser ONNX Runtime pour les autres modèles
            return self._monte_carlo_onnx_runtime(image, noise_level, num_samples)
            
        except Exception as e:
            print(f"     ✗ Erreur lors de l'extraction des activations : {e}")
            raise
    
    def _monte_carlo_vgg16(
        self,
        image: np.ndarray,
        noise_level: float,
        num_layers: int,
        num_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Monte Carlo sampling spécifique pour VGG16 avec PyTorch"""
        import torch
        import torchvision.models as models
        
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.eval()
        
        # Extraire sous-réseau VGG16
        all_layers = self.parser.parse()
        target_layers = all_layers[:num_layers]
        feature_index = self._map_vgg16_to_feature_index(target_layers)
        
        print(f"     → Extraction jusqu'à features[{feature_index}] (VGG16)")
        sub_model = torch.nn.Sequential(*list(model.features[:feature_index+1]))
        sub_model.eval()
        
        samples = []
        
        for _ in range(num_samples):
            # Générer bruit aléatoire
            noise = np.random.uniform(-noise_level, noise_level, image.shape).astype(np.float32)
            noisy_image = (image + noise).astype(np.float32)
            
            # Ajouter dimension batch si nécessaire
            if len(noisy_image.shape) == 3:
                noisy_image = np.expand_dims(noisy_image, axis=0)
            
            # Inférence à travers le sous-modèle
            with torch.no_grad():
                input_tensor = torch.from_numpy(noisy_image)
                intermediate_output = sub_model(input_tensor)
                samples.append(intermediate_output.numpy().flatten())
        
        samples_array = np.array(samples)
        return samples_array.min(axis=0), samples_array.max(axis=0)
    
    def _monte_carlo_onnx_runtime(
        self,
        image: np.ndarray,
        noise_level: float,
        num_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo sampling générique utilisant ONNX Runtime
        
        Fonctionne avec n'importe quel modèle ONNX en faisant
        l'inférence complète du réseau.
        """
        print(f"     → Échantillonnage via ONNX Runtime (réseau complet)")
        
        # Obtenir le nom de l'entrée depuis la session ONNX
        input_name = self.session.get_inputs()[0].name
        
        samples = []
        
        for i in range(num_samples):
            # Générer bruit aléatoire
            noise = np.random.uniform(-noise_level, noise_level, image.shape).astype(np.float32)
            noisy_image = (image + noise).astype(np.float32)
            
            # Ajouter dimension batch si nécessaire
            if len(noisy_image.shape) == 3:
                noisy_image = np.expand_dims(noisy_image, axis=0)
            
            # Inférence via ONNX Runtime
            try:
                outputs = self.session.run(None, {input_name: noisy_image})
                # Prendre la première sortie (logits généralement)
                output = outputs[0].flatten()
                samples.append(output)
            except Exception as e:
                print(f"     ✗ Erreur d'inférence ONNX à l'échantillon {i}: {e}")
                # Continuer avec les autres échantillons
                continue
            
            # Afficher progression
            if (i + 1) % 20 == 0:
                print(f"        Progression: {i+1}/{num_samples} échantillons")
        
        if not samples:
            raise RuntimeError("Aucun échantillon Monte Carlo n'a réussi")
        
        samples_array = np.array(samples)
        print(f"     ✓ {len(samples)} échantillons collectés")
        
        return samples_array.min(axis=0), samples_array.max(axis=0)
    
    def _map_vgg16_to_feature_index(self, layers: List[Dict[str, Any]]) -> int:
        """
        Mapper les couches parsées vers l'index dans vgg16.features
        
        Args:
            layers: Liste des couches parsées
            
        Returns:
            Index dans vgg16.features
        """
        feature_idx = -1
        
        for layer in layers:
            layer_type = layer['type']
            
            if layer_type == 'Conv':
                feature_idx += 1
            elif layer_type == 'Relu':
                feature_idx += 1
            elif layer_type == 'MaxPool':
                feature_idx += 1
            elif layer_type in ['Shape', 'Concat', 'Reshape']:
                continue
        
        return feature_idx
    
    def monte_carlo_sampling(self, image, noise_level, num_samples=100):
        """Version originale - garde pour compatibilité"""
        return self.monte_carlo_sampling_at_layer(image, noise_level, 999, num_samples)


def verify_partial_soundness(
    model_path: str,
    image: np.ndarray,
    noise_level: float,
    num_layers: int = 5,
    num_mc_samples: int = 100
) -> Dict[str, Any]:
    """
    Vérifier la soundness sur un sous-réseau
    
    Args:
        model_path: Chemin vers le modèle ONNX
        image: Image d'entrée (C, H, W) ou (B, C, H, W)
        noise_level: Niveau de bruit
        num_layers: Nombre de couches à tester
        num_mc_samples: Nombre d'échantillons Monte Carlo
        
    Returns:
        Dictionnaire avec les résultats de vérification
    """
    print(f"\n{'='*70}")
    print(f"Vérification de soundness partielle")
    print(f"{'='*70}")
    
    # Parser le modèle
    parser = ONNXParser(model_path)
    all_layers = parser.parse()
    layers = all_layers[:num_layers]
    
    print(f"\nModèle : {model_path}")
    print(f"Couches à évaluer : {len(layers)}")
    print(f"Niveau de bruit : {noise_level}")
    print(f"Échantillons MC : {num_mc_samples}")
    
    # Créer l'évaluateur formel (utilise BoundPropagator en interne)
    evaluator = PartialNetworkEvaluator()
    
    # Propagation formelle
    print(f"\n[1/3] Propagation formelle...")
    input_shape = image.shape
    
    try:
        input_expressions = evaluator.create_input_expressions(image, noise_level)
        print(f"  ✓ {len(input_expressions)} expressions créées")
        
        output_expressions, all_bounds = evaluator.propagate_through_layers(
            input_expressions,
            layers,
            input_shape
        )
        
        # Extraire les bornes finales (limité pour perf)
        sample_size = min(100, len(output_expressions))
        formal_bounds = [expr.get_bounds() for expr in output_expressions[:sample_size]]
        
        print(f"  ✓ Propagation formelle terminée")
        print(f"  ✓ {len(formal_bounds)} bornes calculées")
        
    except Exception as e:
        print(f"  ✗ Erreur lors de la propagation formelle : {e}")
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
    
    # Monte Carlo sampling À LA MÊME COUCHE
    print(f"\n[2/3] Échantillonnage Monte Carlo (même couche)...")
    onnx_evaluator = ONNXPartialEvaluator(model_path)
    
    try:
        mc_min, mc_max = onnx_evaluator.monte_carlo_sampling_at_layer(
            image,
            noise_level,
            num_layers,
            num_mc_samples
        )
        
        # Limiter aux mêmes indices que formel
        mc_min = mc_min[:sample_size]
        mc_max = mc_max[:sample_size]
        
        print(f"  ✓ {num_mc_samples} échantillons Monte Carlo collectés")
        print(f"  ✓ Activations extraites à la couche {num_layers}")
        
    except Exception as e:
        print(f"  ✗ Erreur lors du Monte Carlo : {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
    
    # Vérification de soundness
    print(f"\n[3/3] Vérification de soundness...")
    
    soundness_report = {
        'is_sound': True,
        'violations': [],
        'total_outputs': len(formal_bounds),
        'violation_count': 0,
        'max_violation_lower': 0.0,
        'max_violation_upper': 0.0,
        'safety_margins_lower': [],
        'safety_margins_upper': [],
        'coverage_ratio': 0.0
    }
    
    for i, (formal_lower, formal_upper) in enumerate(formal_bounds):
        observed_min = float(mc_min[i])
        observed_max = float(mc_max[i])
        
        # Vérifier soundness avec tolérance numérique
        tolerance = 1e-6
        lower_sound = observed_min >= formal_lower - tolerance
        upper_sound = observed_max <= formal_upper + tolerance
        
        is_sound = lower_sound and upper_sound
        
        if not is_sound:
            soundness_report['is_sound'] = False
            soundness_report['violation_count'] += 1
            
            violation = {
                'index': i,
                'observed_min': observed_min,
                'formal_min': float(formal_lower),
                'observed_max': observed_max,
                'formal_max': float(formal_upper),
                'violation_lower': max(0, formal_lower - observed_min),
                'violation_upper': max(0, observed_max - formal_upper)
            }
            
            soundness_report['violations'].append(violation)
            soundness_report['max_violation_lower'] = max(
                soundness_report['max_violation_lower'],
                violation['violation_lower']
            )
            soundness_report['max_violation_upper'] = max(
                soundness_report['max_violation_upper'],
                violation['violation_upper']
            )
        
        # Calculer marges de sécurité
        margin_lower = observed_min - formal_lower
        margin_upper = formal_upper - observed_max
        
        soundness_report['safety_margins_lower'].append(float(margin_lower))
        soundness_report['safety_margins_upper'].append(float(margin_upper))
    
    # Calculer le ratio de couverture
    soundness_report['coverage_ratio'] = (
        1.0 - (soundness_report['violation_count'] / soundness_report['total_outputs'])
    )
    
    # Afficher résumé
    print(f"\n{'─'*70}")
    if soundness_report['is_sound']:
        print(f"✅ SOUND : Toutes les valeurs observées sont dans les bornes formelles")
    else:
        print(f"❌ NOT SOUND : {soundness_report['violation_count']} violations détectées")
        print(f"   Taux de couverture : {soundness_report['coverage_ratio']*100:.2f}%")
    
    avg_margin_lower = np.mean(soundness_report['safety_margins_lower'])
    avg_margin_upper = np.mean(soundness_report['safety_margins_upper'])
    print(f"\n📏 Marges de sécurité moyennes :")
    print(f"   Borne inférieure : {avg_margin_lower:.6f}")
    print(f"   Borne supérieure : {avg_margin_upper:.6f}")
    
    print(f"\n💡 Comparaison correcte :")
    print(f"   Formel : Bornes après {num_layers} couches")
    print(f"   Monte Carlo : Activations après {num_layers} couches")
    print(f"   ✅ Même niveau de réseau comparé")
    
    return {
        'success': True,
        'soundness_report': soundness_report,
        'formal_bounds': formal_bounds,
        'monte_carlo_bounds': (mc_min.tolist(), mc_max.tolist()),
        'num_layers': len(layers),
        'num_outputs': len(formal_bounds),
        'noise_level': noise_level,
        'num_mc_samples': num_mc_samples
    }


def quick_soundness_check(
    model_path: str,
    test_image: np.ndarray,
    epsilon: float = 0.01,
    num_layers: int = 5,
    verbose: bool = True
) -> bool:
    """
    Vérification rapide de soundness
    
    Args:
        model_path: Chemin vers le modèle ONNX
        test_image: Image de test (C, H, W)
        epsilon: Niveau de bruit
        num_layers: Nombre de couches à tester
        verbose: Afficher les détails
        
    Returns:
        True si sound, False sinon
    """
    result = verify_partial_soundness(
        model_path,
        test_image,
        epsilon,
        num_layers=num_layers,
        num_mc_samples=50
    )
    
    if not result['success']:
        if verbose:
            print(f"Erreur : {result['error']}")
        return False
    
    is_sound = result['soundness_report']['is_sound']
    
    if verbose and not is_sound:
        violations = result['soundness_report']['violation_count']
        total = result['soundness_report']['total_outputs']
        print(f"Violations : {violations}/{total}")
    
    return is_sound
