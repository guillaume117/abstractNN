"""
Générateur de rapports détaillés sur la propagation à travers le réseau
"""

import json
from typing import List, Dict, Any
from datetime import datetime
from .affine_engine import AffineExpression


class ReportGenerator:
    """Génère des rapports détaillés sur l'analyse formelle"""
    
    def __init__(self):
        self.layer_reports = []
        self.summary = {}
    
    def analyze_expressions(self, expressions: List[AffineExpression], 
                           layer_info: Dict[str, Any]) -> Dict:
        """
        Analyse un ensemble d'expressions affines
        
        Returns:
            Statistiques détaillées sur les expressions
        """
        if not expressions:
            return {
                'num_expressions': 0,
                'avg_symbols_per_expr': 0,
                'max_symbols': 0,
                'min_symbols': 0,
                'total_symbols': 0,
                'avg_bound_width': 0
            }
        
        num_symbols_list = [len(expr.coefficients) for expr in expressions]
        bound_widths = []
        
        for expr in expressions:
            lower, upper = expr.get_bounds()
            bound_widths.append(upper - lower)
        
        stats = {
            'num_expressions': len(expressions),
            'avg_symbols_per_expr': sum(num_symbols_list) / len(num_symbols_list),
            'max_symbols': max(num_symbols_list),
            'min_symbols': min(num_symbols_list),
            'total_symbols': sum(num_symbols_list),
            'avg_bound_width': sum(bound_widths) / len(bound_widths),
            'max_bound_width': max(bound_widths),
            'min_bound_width': min(bound_widths),
            'layer_type': layer_info.get('type', 'Unknown'),
            'layer_name': layer_info.get('name', 'Unknown')
        }
        
        return stats
    
    def add_layer_report(self, expressions: List[AffineExpression],
                        layer_info: Dict[str, Any],
                        execution_time: float = 0.0):
        """Ajoute un rapport de couche"""
        stats = self.analyze_expressions(expressions, layer_info)
        stats['execution_time'] = execution_time
        self.layer_reports.append(stats)
    
    def generate_summary(self) -> Dict:
        """Génère un résumé global de l'analyse"""
        if not self.layer_reports:
            return {}
        
        total_time = sum(r['execution_time'] for r in self.layer_reports)
        total_expressions = sum(r['num_expressions'] for r in self.layer_reports)
        total_symbols = sum(r['total_symbols'] for r in self.layer_reports)
        
        # Compte par type de couche
        layer_types = {}
        for report in self.layer_reports:
            layer_type = report['layer_type']
            if layer_type not in layer_types:
                layer_types[layer_type] = {
                    'count': 0,
                    'total_time': 0.0,
                    'total_expressions': 0
                }
            layer_types[layer_type]['count'] += 1
            layer_types[layer_type]['total_time'] += report['execution_time']
            layer_types[layer_type]['total_expressions'] += report['num_expressions']
        
        self.summary = {
            'total_layers': len(self.layer_reports),
            'total_execution_time': total_time,
            'total_expressions_processed': total_expressions,
            'total_symbols': total_symbols,
            'avg_symbols_per_layer': total_symbols / len(self.layer_reports),
            'layer_types': layer_types,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.summary
    
    def print_detailed_report(self):
        """Affiche un rapport détaillé sur la console"""
        print("\n" + "="*80)
        print("RAPPORT DÉTAILLÉ - PROPAGATION DES EXPRESSIONS AFFINES")
        print("="*80)
        
        print(f"\n{'Couche':<20} | {'Type':<15} | {'Nb Expr':<10} | {'Moy Symb':<12} | {'Max Symb':<10} | {'Temps (ms)':<12}")
        print("-"*100)
        
        for i, report in enumerate(self.layer_reports):
            print(f"{report['layer_name']:<20} | "
                  f"{report['layer_type']:<15} | "
                  f"{report['num_expressions']:<10} | "
                  f"{report['avg_symbols_per_expr']:<12.2f} | "
                  f"{report['max_symbols']:<10} | "
                  f"{report['execution_time']*1000:<12.3f}")
        
        # Résumé
        if self.summary:
            print("\n" + "="*80)
            print("RÉSUMÉ GLOBAL")
            print("="*80)
            
            print(f"\nNombre total de couches: {self.summary['total_layers']}")
            print(f"Temps d'exécution total: {self.summary['total_execution_time']:.3f}s")
            print(f"Expressions totales traitées: {self.summary['total_expressions_processed']}")
            print(f"Symboles totaux: {self.summary['total_symbols']}")
            print(f"Moyenne symboles par couche: {self.summary['avg_symbols_per_layer']:.2f}")
            
            print("\n" + "-"*80)
            print("STATISTIQUES PAR TYPE DE COUCHE")
            print("-"*80)
            print(f"\n{'Type':<20} | {'Nombre':<10} | {'Temps total (ms)':<18} | {'Expressions':<15}")
            print("-"*80)
            
            for layer_type, stats in self.summary['layer_types'].items():
                print(f"{layer_type:<20} | "
                      f"{stats['count']:<10} | "
                      f"{stats['total_time']*1000:<18.3f} | "
                      f"{stats['total_expressions']:<15}")
        
        print("\n" + "="*80)
    
    def print_complexity_analysis(self):
        """Analyse la complexité symbolique du réseau"""
        print("\n" + "="*80)
        print("ANALYSE DE COMPLEXITÉ SYMBOLIQUE")
        print("="*80)
        
        print("\nÉvolution de la complexité symbolique:")
        print(f"\n{'Couche':<20} | {'Nb Symboles (moy)':<20} | {'Largeur bornes (moy)':<25}")
        print("-"*80)
        
        for report in self.layer_reports:
            print(f"{report['layer_name']:<20} | "
                  f"{report['avg_symbols_per_expr']:<20.2f} | "
                  f"{report['avg_bound_width']:<25.6f}")
        
        # Détecte les couches critiques
        print("\n" + "-"*80)
        print("COUCHES CRITIQUES")
        print("-"*80)
        
        # Couche avec le plus de symboles
        max_symbols_layer = max(self.layer_reports, key=lambda r: r['max_symbols'])
        print(f"\nCouche avec le plus de symboles:")
        print(f"  {max_symbols_layer['layer_name']} ({max_symbols_layer['layer_type']})")
        print(f"  Max symboles: {max_symbols_layer['max_symbols']}")
        
        # Couche avec les bornes les plus larges
        max_width_layer = max(self.layer_reports, key=lambda r: r['avg_bound_width'])
        print(f"\nCouche avec les bornes les plus larges:")
        print(f"  {max_width_layer['layer_name']} ({max_width_layer['layer_type']})")
        print(f"  Largeur moyenne: {max_width_layer['avg_bound_width']:.6f}")
        
        # Couche la plus lente
        slowest_layer = max(self.layer_reports, key=lambda r: r['execution_time'])
        print(f"\nCouche la plus lente:")
        print(f"  {slowest_layer['layer_name']} ({slowest_layer['layer_type']})")
        print(f"  Temps: {slowest_layer['execution_time']*1000:.3f}ms")
        
        print("\n" + "="*80)
    
    def export_to_json(self, filepath: str):
        """Exporte le rapport au format JSON"""
        report_data = {
            'summary': self.summary,
            'layer_reports': self.layer_reports
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nRapport exporté vers: {filepath}")
    
    def export_to_csv(self, filepath: str):
        """Exporte le rapport au format CSV"""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            fieldnames = ['layer_name', 'layer_type', 'num_expressions', 
                         'avg_symbols_per_expr', 'max_symbols', 'min_symbols',
                         'total_symbols', 'avg_bound_width', 'execution_time']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for report in self.layer_reports:
                writer.writerow({k: report.get(k, '') for k in fieldnames})
        
        print(f"Rapport CSV exporté vers: {filepath}")
