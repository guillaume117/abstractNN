"""
Parser pour modèles ONNX
"""

import onnx
import onnxruntime as ort
import numpy as np
from typing import List, Dict, Any


class ONNXParser:
    """Parse et extrait les opérations d'un modèle ONNX"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.session = ort.InferenceSession(model_path)
        self.graph = self.model.graph
        self.layers = []
        
    def parse(self) -> List[Dict[str, Any]]:
        """Parse le graphe ONNX et extrait les couches"""
        self.layers = []
        
        # Extrait les poids et biais
        initializers = {init.name: self._tensor_to_array(init) 
                       for init in self.graph.initializer}
        
        for node in self.graph.node:
            layer_info = {
                'type': node.op_type,
                'name': node.name,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': self._parse_attributes(node.attribute)
            }
            
            # Ajoute les poids et biais si disponibles
            if node.op_type in ['Conv']:
                # Pour Conv: inputs[0]=data, inputs[1]=weight, inputs[2]=bias (optionnel)
                if len(node.input) >= 2 and node.input[1] in initializers:
                    layer_info['weights'] = initializers[node.input[1]]
                if len(node.input) >= 3 and node.input[2] in initializers:
                    layer_info['bias'] = initializers[node.input[2]]
            
            elif node.op_type in ['MatMul', 'Gemm']:
                # Pour MatMul/Gemm: inputs[0]=data, inputs[1]=weight
                if len(node.input) >= 2 and node.input[1] in initializers:
                    weights = initializers[node.input[1]]
                    # Transpose si nécessaire pour MatMul
                    if node.op_type == 'MatMul' and len(weights.shape) == 2:
                        weights = weights.T
                    layer_info['weights'] = weights
                
                # Pour Gemm, le biais peut être dans inputs[2]
                if node.op_type == 'Gemm' and len(node.input) >= 3 and node.input[2] in initializers:
                    layer_info['bias'] = initializers[node.input[2]]
            
            self.layers.append(layer_info)
        
        return self.layers
    
    def _tensor_to_array(self, tensor) -> np.ndarray:
        """Convertit un tensor ONNX en numpy array"""
        return onnx.numpy_helper.to_array(tensor)
    
    def _parse_attributes(self, attributes) -> Dict[str, Any]:
        """Parse les attributs d'un nœud ONNX"""
        attrs = {}
        for attr in attributes:
            if attr.HasField('f'):
                attrs[attr.name] = attr.f
            elif attr.HasField('i'):
                attrs[attr.name] = attr.i
            elif attr.HasField('s'):
                attrs[attr.name] = attr.s.decode('utf-8')
            elif attr.ints:
                attrs[attr.name] = list(attr.ints)
        return attrs
    
    def get_input_shape(self) -> tuple:
        """Retourne la forme de l'entrée du modèle"""
        input_info = self.session.get_inputs()[0]
        return input_info.shape
    
    def get_output_names(self) -> List[str]:
        """Retourne les noms des sorties"""
        return [output.name for output in self.session.get_outputs()]
