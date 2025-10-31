"""
Agrégation des résultats et identification des classes robustes
"""

from typing import List, Dict, Tuple, Optional
from .affine_engine import AffineExpression


class ResultAggregator:
    """Agrège les résultats et identifie les classes robustes"""
    
    @staticmethod
    def compute_class_bounds(output_expressions: List[AffineExpression],
                            class_names: List[str] = None) -> Dict[str, Tuple[float, float]]:
        """Calcule les bornes pour chaque classe de sortie"""
        if class_names is None:
            class_names = [f"class_{i}" for i in range(len(output_expressions))]
        
        bounds = {}
        for i, expr in enumerate(output_expressions):
            lower, upper = expr.get_bounds()
            bounds[class_names[i]] = (lower, upper)
        
        return bounds
    
    @staticmethod
    def find_robust_class(bounds: Dict[str, Tuple[float, float]]) -> Optional[str]:
        """
        Trouve la classe robuste: celle dont la borne inférieure
        est supérieure aux bornes supérieures de toutes les autres
        """
        for candidate_class, (cand_lower, cand_upper) in bounds.items():
            is_robust = True
            
            for other_class, (other_lower, other_upper) in bounds.items():
                if candidate_class == other_class:
                    continue
                
                # La borne inférieure du candidat doit être > borne sup des autres
                if cand_lower <= other_upper:
                    is_robust = False
                    break
            
            if is_robust:
                return candidate_class
        
        return None
    
    @staticmethod
    def compute_robustness_margin(bounds: Dict[str, Tuple[float, float]],
                                 predicted_class: str) -> float:
        """
        Calcule la marge de robustesse pour une classe prédite
        """
        pred_lower, _ = bounds[predicted_class]
        
        max_other_upper = max(
            upper for cls, (_, upper) in bounds.items() 
            if cls != predicted_class
        )
        
        return pred_lower - max_other_upper
