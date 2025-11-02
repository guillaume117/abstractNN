

Affine Engine
=============

The affine engine is the core component for symbolic expression management and propagation.

.. automodule:: abstractnn.affine_engine
   :members:
   :undoc-members:
   :show-inheritance:

AffineExpression
----------------

.. autoclass:: abstractnn.affine_engine.AffineExpression
   :members:
   :special-members: __init__, __add__, __mul__
   :show-inheritance:

   .. attribute:: constant
      
      The constant term :math:`c_0` in the affine expression

   .. attribute:: coefficients
      
      Dictionary mapping symbol indices to coefficients

   .. attribute:: bounds
      
      Dictionary mapping symbol indices to their bounds :math:`[\underline{\epsilon_i}, \overline{\epsilon_i}]`

   .. method:: get_bounds()
      
      Compute interval bounds for this expression:
      
      .. math::
         
         \underline{y} = c_0 + \sum_{i: c_i \geq 0} c_i \cdot \underline{\epsilon_i} + \sum_{i: c_i < 0} c_i \cdot \overline{\epsilon_i}
         
         \overline{y} = c_0 + \sum_{i: c_i \geq 0} c_i \cdot \overline{\epsilon_i} + \sum_{i: c_i < 0} c_i \cdot \underline{\epsilon_i}

AffineExpressionEngine
----------------------

.. autoclass:: abstractnn.affine_engine.AffineExpressionEngine
   :members:
   :undoc-members:
   :show-inheritance:

   .. method:: create_input_expressions(image, noise_level)
      
      Create affine expressions for perturbed input:
      
      .. math::
         
         x_i = \text{image}_i + \epsilon \cdot \delta_i, \quad \delta_i \in [-1, 1]
      
      :param numpy.ndarray image: Input image
      :param float noise_level: Perturbation radius :math:`\epsilon`
      :return: List of affine expressions
      :rtype: List[AffineExpression]

   .. method:: conv2d_layer(expressions, weights, bias, input_shape, stride, padding, dilation)
      
      Propagate through convolutional layer:
      
      .. math::
         
         y_{out}[n, c_{out}, h, w] = \text{bias}[c_{out}] + \sum_{c_{in}=0}^{C_{in}-1} \sum_{k_h, k_w} \text{weight}[c_{out}, c_{in}, k_h, k_w] \cdot x_{in}[n, c_{in}, h \cdot s_h + k_h, w \cdot s_w + k_w]

   .. method:: linear_layer(expressions, weights, bias)
      
      Propagate through fully-connected layer:
      
      .. math::
         
         y = Wx + b

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from abstractnn..affine_engine import AffineExpressionEngine
    import numpy as np

    engine = AffineExpressionEngine()

    # Create perturbed input
    image = np.random.rand(3, 28, 28).astype(np.float32)
    expressions = engine.create_input_expressions(image, noise_level=0.1)

    print(f"Created {len(expressions)} expressions")
    print(f"First expression bounds: {expressions[0].get_bounds()}")

Convolution Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Define conv layer
    weights = np.random.randn(64, 3, 3, 3).astype(np.float32)
    bias = np.random.randn(64).astype(np.float32)

    # Propagate
    output_exprs, output_shape = engine.conv2d_layer(
        expressions,
        weights,
        bias,
        input_shape=(1, 3, 28, 28),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1)
    )

    print(f"Output shape: {output_shape}")
    print(f"Output expressions: {len(output_exprs)}")