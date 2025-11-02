
Bound Propagator
================

The bound propagator orchestrates layer-by-layer formal verification through neural networks.

.. automodule:: abstractnn.bound_propagator
   :members:
   :undoc-members:
   :show-inheritance:

BoundPropagator
---------------

.. autoclass:: abstractnn.bound_propagator.BoundPropagator
   :members:
   :special-members: __init__
   :show-inheritance:

   Main class for propagating bounds through neural networks.

   .. method:: propagate(input_expressions, layers, input_shape)
      
      Propagate affine expressions through network layers:
      
      .. math::
         
         \mathbf{y} = f_L \circ f_{L-1} \circ \cdots \circ f_1(\mathbf{x})
      
      where each :math:`f_i` is a layer transformation.
      
      :param List[AffineExpression] input_expressions: Input symbolic expressions
      :param List[Dict] layers: Network layers from ONNX parser
      :param Tuple input_shape: Shape of input (B, C, H, W)
      :return: Output expressions after full propagation
      :rtype: List[AffineExpression]

   .. method:: _propagate_layer(expressions, layer, current_shape)
      
      Propagate through a single layer.
      
      Supports:
      
      - **Linear**: Conv, Gemm/Linear, BatchNorm
      - **Non-linear**: ReLU, MaxPool
      - **Utility**: Reshape, Flatten, Concat

Layer Handlers
--------------

Conv2d Handler
~~~~~~~~~~~~~~

.. code-block:: python

    def _handle_conv(self, expressions, layer, input_shape):
        """
        Handle convolutional layer.
        
        Forward pass:
            y[n, c_out, h, w] = bias[c_out] + 
                Σ weight[c_out, c_in, k_h, k_w] * 
                  x[n, c_in, h*s_h + k_h, w*s_w + k_w]
        
        Symbolic:
            Each output neuron is a weighted sum of input expressions
        """

ReLU Handler
~~~~~~~~~~~~

.. code-block:: python

    def _handle_relu(self, expressions):
        """
        Handle ReLU activation with relaxation.
        
        Cases:
        1. l >= 0: y = x (active)
        2. u <= 0: y = 0 (inactive)
        3. l < 0 < u: y ∈ [0, max(|l|, u)] (ambiguous, needs relaxation)
        """

MaxPool Handler
~~~~~~~~~~~~~~~

.. code-block:: python

    def _handle_maxpool(self, expressions, layer, input_shape):
        """
        Handle max pooling conservatively.
        
        For region R: max(R) ≤ max(upper_bounds(R))
        
        Takes union of all possible maxima in pooling window.
        """

Usage Examples
--------------

Basic Propagation
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from abstractnn.affine_engine import AffineExpressionEngine
    from abstractnn.bound_propagator import BoundPropagator
    from abstractnn.onnx_parser import ONNXParser
    
    # Load model
    parser = ONNXParser('model.onnx')
    layers = parser.parse()
    
    # Create propagator
    engine = AffineExpressionEngine()
    propagator = BoundPropagator(engine, enable_reporting=True)
    
    # Create input
    import numpy as np
    image = np.random.rand(3, 28, 28).astype(np.float32)
    input_exprs = engine.create_input_expressions(image, noise_level=0.1)
    
    # Propagate
    output_exprs = propagator.propagate(
        input_exprs,
        layers,
        input_shape=(1, 3, 28, 28)
    )
    
    # Get bounds
    bounds = [expr.get_bounds() for expr in output_exprs]
    print(f"Output bounds: {bounds[0]}")

With Checkpointing
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Enable checkpointing for large models
    propagator = BoundPropagator(
        engine,
        enable_reporting=True,
        checkpoint_frequency=5  # Save every 5 layers
    )
    
    output_exprs = propagator.propagate(
        input_exprs,
        layers,
        input_shape=(1, 3, 224, 224)
    )
    
    # Access intermediate checkpoints
    checkpoints = propagator.get_checkpoints()
    for layer_idx, checkpoint in checkpoints.items():
        print(f"Layer {layer_idx}: {len(checkpoint)} expressions")

Layer-by-Layer Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Analyze bound growth through layers
    propagator = BoundPropagator(engine, enable_reporting=True)
    
    current_exprs = input_exprs
    bound_widths = []
    
    for i, layer in enumerate(layers):
        current_exprs = propagator._propagate_layer(
            current_exprs,
            layer,
            current_shape
        )
        
        # Measure bound width
        bounds = [e.get_bounds() for e in current_exprs[:10]]
        avg_width = np.mean([u - l for l, u in bounds])
        bound_widths.append(avg_width)
        
        print(f"Layer {i} ({layer['type']}): avg width = {avg_width:.4f}")
    
    # Plot bound growth
    import matplotlib.pyplot as plt
    plt.plot(bound_widths)
    plt.xlabel('Layer')
    plt.ylabel('Average Bound Width')
    plt.title('Bound Relaxation Through Network')
    plt.show()



- :doc:`affine_engine`: Affine expression management
- :doc:`relaxer`: Non-linear relaxation techniques
- :doc:`../user_guide/propagation`: Propagation user guide