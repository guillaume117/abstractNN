Overview
========

What is AbstractNN?
------------------

AbstractNN is a **formal verification framework** for deep neural networks that uses:

- **Abstract Interpretation**: Sound over-approximation of network behavior
- **Affine Arithmetic**: Tracking input-output relationships symbolically  
- **Bound Propagation**: Computing reachable output sets

Key Features
------------

Sound Verification
~~~~~~~~~~~~~~~~~~

AbstractNN provides **mathematically sound** guarantees:

.. math::

   \forall x \in [x_0 - \epsilon, x_0 + \epsilon], \quad f(x) \in [\underline{y}, \overline{y}]

where :math:`[\underline{y}, \overline{y}]` are the computed output bounds.

Affine Expression Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each neuron's output is represented as an affine expression:

.. math::

   y = c_0 + \sum_{i=1}^{n} c_i \cdot \epsilon_i

where:

- :math:`c_0` is the constant term (center point)
- :math:`c_i` are coefficients representing dependencies
- :math:`\epsilon_i \in [-1, 1]` are symbolic noise variables

Supported Operations
~~~~~~~~~~~~~~~~~~~~

**Linear Layers:**

- Convolution (Conv2d)
- Fully Connected (Linear/Gemm)
- Batch Normalization

**Non-linear Layers:**

- ReLU (with linear relaxation)
- MaxPool (conservative approximation)
- Sigmoid/Tanh (interval relaxation)

**Utility Operations:**

- Flatten/Reshape
- Concatenation
- Element-wise operations

Architecture
------------

.. code-block:: text

    ┌─────────────────────────────────────────────┐
    │          User Interface                     │
    ├─────────────────────────────────────────────┤
    │  ONNX Parser  │  Partial Evaluator          │
    ├─────────────────────────────────────────────┤
    │  Bound        │  Relaxer    │  Soundness    │
    │  Propagator   │             │  Checker      │
    ├─────────────────────────────────────────────┤
    │         Affine Expression Engine            │
    └─────────────────────────────────────────────┘

Core Components
---------------

1. **Affine Engine** (:class:`~modules.affine_engine.AffineExpressionEngine`)
   
   - Manages symbolic expressions
   - Propagates through linear layers
   - Computes bounds efficiently

2. **Bound Propagator** (:class:`~modules.bound_propagator.BoundPropagator`)
   
   - Orchestrates layer-by-layer propagation
   - Handles layer type dispatch
   - Maintains symbolic state

3. **Relaxer** (:class:`~modules.relaxer.NonLinearRelaxer`)
   
   - Approximates non-linear activations
   - Provides sound over-approximations
   - Multiple relaxation strategies

4. **Soundness Checker** (:class:`~modules.soundness_checker.SoundnessChecker`)
   
   - Validates computed bounds
   - Monte Carlo comparison
   - Violation detection

Use Cases
---------

Adversarial Robustness Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify that a network's prediction remains stable under :math:`L_\infty` perturbations:

.. code-block:: python

    # Check if prediction is robust to ε=0.01 perturbations
    from modules.partial_evaluator import verify_partial_soundness
    
    result = verify_partial_soundness(
        model_path='vgg16.onnx',
        image=test_image,
        noise_level=0.01,
        num_layers=10
    )
    
    is_robust = result['soundness_report']['is_sound']

Certified Defense Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate certified accuracy of defended models:

.. code-block:: python

    certified_correct = 0
    for image, label in test_set:
        result = verify_partial_soundness(
            model_path='robust_model.onnx',
            image=image,
            noise_level=0.03,
            num_layers=999  # Full network
        )
        
        if result['soundness_report']['is_sound']:
            certified_correct += 1
    
    certified_acc = certified_correct / len(test_set)

Safety-Critical Systems
~~~~~~~~~~~~~~~~~~~~~~~

Verify safety properties for deployment:

.. code-block:: python

    # Verify output stays within safe bounds
    result = verify_partial_soundness(
        model_path='autopilot.onnx',
        image=sensor_input,
        noise_level=sensor_noise
    )
    
    bounds = result['formal_bounds']
    is_safe = all(l >= safe_min and u <= safe_max 
                  for l, u in bounds)

Comparison with Other Methods
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Method
     - Soundness
     - Tightness
     - Scalability
     - Speed
   * - Monte Carlo
     - ❌ No
     - N/A
     - ✅ High
     - ✅ Fast
   * - MILP
     - ✅ Yes
     - ✅ Exact
     - ❌ Low
     - ❌ Slow
   * - Interval
     - ✅ Yes
     - ❌ Loose
     - ✅ High
     - ✅ Fast
   * - **AbstracNN**
     - ✅ Yes
     - ⚠️ Good
     - ⚠️ Medium
     - ⚠️ Medium

Limitations
-----------

Memory Requirements
~~~~~~~~~~~~~~~~~~~

For large networks like VGG16:

- Full symbolic propagation: ~168 GB (150,528 symbols)
- Partial network evaluation: Feasible for 3-5 layers
- Recommended: Use reduced resolution or GPU

Precision vs Soundness Tradeoff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Over-approximation may be conservative
- Relaxations accumulate through layers
- Deep networks require careful optimization

Supported Architectures
~~~~~~~~~~~~~~~~~~~~~~~~

Currently optimized for:

- CNNs (VGG, ResNet-style)
- Feedforward networks
- Limited RNN support

Next Steps
----------

- :doc:`installation`: Install abstractNN
- :doc:`quickstart`: Run your first verification
- :doc:`tutorials/index`: Detailed tutorials