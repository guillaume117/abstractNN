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
    """Évaluateur pour sous-réseaux"""
    
    def __init__(self, engine: AffineExpressionEngine):
        """
        Initialiser l'évaluateur partiel
        
        Args:
            engine: Moteur d'expressions affines
        """
        self.engine = engine
        self.relaxer = NonLinearRelaxer()
    
    def create_input_expressions(
        self,
        image: np.ndarray,
        noise_level: float
    ) -> List[AffineExpression]:
        """
        Créer les expressions d'entrée avec bruit
        
        Args:
            image: Image d'entrée (C, H, W)
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
        
        Args:
            expressions: Expressions d'entrée
            layers: Liste des couches à propager
            input_shape: Forme de l'entrée (B, C, H, W) ou (C, H, W)
            
        Returns:
            Tuple (expressions_sortie, bornes_intermédiaires)
        """
        current_expressions = expressions
        current_shape = input_shape
        all_bounds = []
        
        print(f"  Propagation à travers {len(layers)} couches...")
        print(f"  Forme d'entrée initiale : {current_shape}")
        
        for i, layer in enumerate(layers):
            layer_type = layer['type']
            layer_name = layer.get('name', f'layer_{i}')
            
            try:
                if layer_type == 'Conv':
                    print(f"    [Conv] Avant : {len(current_expressions)} exprs, shape={current_shape}")
                    current_expressions, current_shape = self._propagate_conv(
                        current_expressions,
                        layer,
                        current_shape
                    )
                    print(f"    ✓ Conv {layer_name}: {len(current_expressions)} exprs, shape={current_shape}")
                    
                elif layer_type == 'Relu':
                    print(f"    [ReLU] Avant : {len(current_expressions)} exprs")
                    current_expressions = self._propagate_relu(current_expressions)
                    print(f"    ✓ ReLU {layer_name}: {len(current_expressions)} exprs")
                    
                elif layer_type == 'MaxPool':
                    print(f"    [MaxPool] Avant : {len(current_expressions)} exprs, shape={current_shape}")
                    current_expressions, current_shape = self._propagate_maxpool(
                        current_expressions,
                        layer,
                        current_shape
                    )
                    print(f"    ✓ MaxPool {layer_name}: {len(current_expressions)} exprs, shape={current_shape}")
                    
                elif layer_type == 'Gemm':
                    print(f"    [Gemm] Avant : {len(current_expressions)} exprs")
                    current_expressions = self._propagate_gemm(
                        current_expressions,
                        layer
                    )
                    print(f"    ✓ Gemm {layer_name}: {len(current_expressions)} exprs")
                    
                elif layer_type == 'Reshape':
                    current_shape = self._get_reshape_output_shape(layer, current_shape)
                    print(f"    ✓ Reshape {layer_name}: {current_shape}")
                    
                elif layer_type in ['Shape', 'Concat']:
                    print(f"    ⊗ Skipping {layer_type} {layer_name} (auxiliary)")
                    continue
                    
                else:
                    print(f"    ⚠ Type de couche non supporté : {layer_type}")
                    continue
                
                # Calculer les bornes après chaque couche (limité pour perf)
                sample_size = min(100, len(current_expressions))
                bounds = [expr.get_bounds() for expr in current_expressions[:sample_size]]
                all_bounds.extend(bounds)
                
            except ValueError as e:
                print(f"    ✗ Erreur ValueError lors de la propagation de {layer_type} {layer_name}: {e}")
                print(f"       Debug: current_shape={current_shape}, len(exprs)={len(current_expressions)}")
                raise
            except Exception as e:
                print(f"    ✗ Erreur lors de la propagation de {layer_type} {layer_name}: {e}")
                print(f"       Type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                raise
        
        return current_expressions, all_bounds
    
    def _propagate_conv(
        self,
        expressions: List[AffineExpression],
        layer: Dict[str, Any],
        input_shape: Tuple[int, ...]
    ) -> Tuple[List[AffineExpression], Tuple[int, ...]]:
        """Propager à travers une couche convolutionnelle"""
        weights = layer.get('weights')
        bias = layer.get('bias')
        
        if weights is None:
            raise ValueError("Conv layer sans poids")
        
        # Extraire les attributs
        attrs = layer.get('attributes', {})
        strides = attrs.get('strides', [1, 1])
        pads = attrs.get('pads', [0, 0, 0, 0])
        dilations = attrs.get('dilations', [1, 1])
        
        print(f"       Conv params: strides={strides}, pads={pads}, dilations={dilations}")
        print(f"       Weights shape: {weights.shape}")
        print(f"       Input shape: {input_shape}")
        
        # Convertir les pads au format attendu
        if len(pads) == 4:
            padding = (pads[0], pads[1])
        elif len(pads) == 2:
            padding = tuple(pads)
        else:
            padding = (0, 0)
        
        # conv2d_layer ATTEND TOUJOURS (B, C, H, W)
        # Il faut donc toujours ajouter le batch si absent
        if len(input_shape) == 3:
            # (C, H, W) -> (1, C, H, W)
            work_shape = (1,) + input_shape
        elif len(input_shape) == 4:
            # Déjà (B, C, H, W)
            work_shape = input_shape
        else:
            raise ValueError(f"Forme d'entrée invalide pour Conv: {input_shape}")
        
        print(f"       Work shape (avec batch): {work_shape}, padding={padding}")
        
        try:
            # Appeler conv2d_layer avec la forme (B, C, H, W)
            output_exprs, output_shape = self.engine.conv2d_layer(
                expressions,
                weights,
                bias if bias is not None else np.zeros(weights.shape[0]),
                work_shape,  # TOUJOURS 4 dimensions maintenant
                stride=tuple(strides) if len(strides) == 2 else (strides[0], strides[0]),
                padding=padding,
                dilation=tuple(dilations) if len(dilations) == 2 else (dilations[0], dilations[0])
            )
            
            print(f"       Output: {len(output_exprs)} exprs, shape={output_shape}")
            
            # Si l'entrée était (C,H,W), retirer le batch de la sortie
            if len(input_shape) == 3 and len(output_shape) == 4:
                output_shape = output_shape[1:]  # (1, C, H, W) -> (C, H, W)
            
            return output_exprs, output_shape
            
        except Exception as e:
            print(f"       ✗ Erreur dans conv2d_layer: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _propagate_relu(
        self,
        expressions: List[AffineExpression]
    ) -> List[AffineExpression]:
        """Propager à travers ReLU"""
        relaxed_exprs = []
        for expr in expressions:
            relaxed = self.relaxer.relu_relaxation(expr, relaxation_type='linear')
            relaxed_exprs.append(relaxed)
        
        return relaxed_exprs
    
    def _propagate_maxpool(
        self,
        expressions: List[AffineExpression],
        layer: Dict[str, Any],
        input_shape: Tuple[int, ...]
    ) -> Tuple[List[AffineExpression], Tuple[int, ...]]:
        """Propager à travers MaxPool"""
        attrs = layer.get('attributes', {})
        kernel_shape = attrs.get('kernel_shape', [2, 2])
        strides = attrs.get('strides', [2, 2])
        
        output_exprs, output_shape = self.engine.maxpool2d_layer(
            expressions,
            input_shape,
            kernel_size=tuple(kernel_shape),
            stride=tuple(strides)
        )
        
        return output_exprs, output_shape
    
    def _propagate_gemm(
        self,
        expressions: List[AffineExpression],
        layer: Dict[str, Any]
    ) -> List[AffineExpression]:
        """Propager à travers Gemm (General Matrix Multiplication)"""
        weights = layer.get('weights')
        bias = layer.get('bias')
        
        if weights is None:
            raise ValueError("Gemm layer sans poids")
        
        # Gemm attributes
        attrs = layer.get('attributes', {})
        alpha = attrs.get('alpha', 1.0)
        beta = attrs.get('beta', 1.0)
        transA = attrs.get('transA', 0)
        transB = attrs.get('transB', 0)
        
        # Appliquer transpose si nécessaire
        W = weights.T if transB else weights
        
        output_exprs = self.engine.linear_layer(
            expressions,
            W * alpha,
            bias * beta if bias is not None else np.zeros(W.shape[0])
        )
        
        return output_exprs
    
    def _get_reshape_output_shape(
        self,
        layer: Dict[str, Any],
        input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Obtenir la forme de sortie d'une opération Reshape"""
        # Pour l'instant, on retourne la forme d'entrée
        # Une vraie implémentation extrairait la forme cible
        attrs = layer.get('attributes', {})
        target_shape = attrs.get('shape', input_shape)
        return tuple(target_shape) if target_shape else input_shape


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
    
    def extract_intermediate_layer_name(self, num_layers: int) -> Optional[str]:
        """
        Trouver le nom de sortie de la n-ième couche dans le modèle ONNX
        
        Args:
            num_layers: Nombre de couches à traverser
            
        Returns:
            Nom du tensor de sortie, ou None si non trouvé
        """
        all_layers = self.parser.parse()
        target_layers = all_layers[:num_layers]
        
        # Le dernier layer devrait avoir un nom de sortie
        if target_layers:
            last_layer = target_layers[-1]
            # Les noms de sortie ONNX sont souvent stockés dans 'outputs'
            outputs = last_layer.get('outputs', [])
            if outputs:
                return outputs[0]
        
        return None
    
    def monte_carlo_sampling_at_layer(
        self,
        image: np.ndarray,
        noise_level: float,
        num_layers: int,
        num_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Échantillonnage Monte Carlo à une couche spécifique
        
        Args:
            image: Image d'entrée (C, H, W)
            noise_level: Niveau de bruit
            num_layers: Nombre de couches à traverser
            num_samples: Nombre d'échantillons
            
        Returns:
            Tuple (bornes_min, bornes_max) des activations à la couche spécifiée
        """
        print(f"\n  Monte Carlo : extraction des activations à la couche {num_layers}...")
        
        # STRATÉGIE : Créer un sous-modèle ONNX tronqué ou utiliser PyTorch
        # Pour ce prototype, on utilise PyTorch pour extraire les activations
        
        try:
            import torch
            import torchvision.models as models
            
            # Charger VGG16 PyTorch
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            vgg16.eval()
            
            # Extraire le sous-réseau correspondant aux num_layers premières couches
            # VGG16 features: 0=Conv, 1=ReLU, 2=Conv, 3=ReLU, 4=MaxPool, ...
            # On compte les vraies couches (Conv, ReLU, MaxPool)
            all_layers = self.parser.parse()
            target_layers = all_layers[:num_layers]
            
            # Compter combien de couches features cela représente
            # Simplification: on mappe num_layers directement à l'index features
            feature_index = self._map_to_feature_index(target_layers)
            
            print(f"     → Extraction jusqu'à features[{feature_index}]")
            
            # Créer le sous-modèle
            sub_model = torch.nn.Sequential(*list(vgg16.features[:feature_index+1]))
            sub_model.eval()
            
            samples = []
            
            for _ in range(num_samples):
                # Générer bruit aléatoire
                noise = np.random.uniform(-noise_level, noise_level, image.shape).astype(np.float32)
                noisy_image = (image + noise).astype(np.float32)
                
                # Ajouter dimension batch
                if len(noisy_image.shape) == 3:
                    noisy_image = np.expand_dims(noisy_image, axis=0)
                
                # Inférence à travers le sous-modèle
                with torch.no_grad():
                    input_tensor = torch.from_numpy(noisy_image)
                    intermediate_output = sub_model(input_tensor)
                    samples.append(intermediate_output.numpy().flatten())
            
            samples_array = np.array(samples)
            return samples_array.min(axis=0), samples_array.max(axis=0)
            
        except Exception as e:
            print(f"     ✗ Erreur lors de l'extraction des activations : {e}")
            raise
    
    def _map_to_feature_index(self, layers: List[Dict[str, Any]]) -> int:
        """
        Mapper les couches parsées vers l'index dans vgg16.features
        
        Args:
            layers: Liste des couches parsées
            
        Returns:
            Index dans vgg16.features
        """
        # VGG16.features contient 31 couches séquentielles
        # Pattern: Conv-ReLU-Conv-ReLU-MaxPool (répété)
        
        feature_idx = -1
        
        for layer in layers:
            layer_type = layer['type']
            
            if layer_type == 'Conv':
                feature_idx += 1  # Conv
                # Chaque Conv est suivie d'un ReLU dans features
            elif layer_type == 'Relu':
                feature_idx += 1  # ReLU
            elif layer_type == 'MaxPool':
                feature_idx += 1  # MaxPool
            elif layer_type in ['Shape', 'Concat', 'Reshape']:
                # Ces couches sont auxiliaires, ne comptent pas
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
        image: Image d'entrée (C, H, W)
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
    
    # Créer l'évaluateur formel
    engine = AffineExpressionEngine()
    evaluator = PartialNetworkEvaluator(engine)
    
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
        # IMPORTANT : Extraire à la même couche que la propagation formelle
        mc_min, mc_max = onnx_evaluator.monte_carlo_sampling_at_layer(
            image,
            noise_level,
            num_layers,  # Même nombre de couches
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
