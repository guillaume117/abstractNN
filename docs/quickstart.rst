

Quick Start Guide
=================

This guide will get you started with AbstractNN in 5 minutes.

Basic Verification
------------------

Simple MNIST Example
~~~~~~~~~~~~~~~~~~~~

Verify a simple MNIST classifier:

.. code-block:: python

    from abstractnn.affine_engine import AffineExpressionEngine
    from abstractnn.onnx_parser import ONNXParser
    from abstractnn.bound_propagator import BoundPropagator
    import numpy as np

    # Load model
    parser = ONNXParser('mnist_model.onnx')
    layers = parser.parse()

    # Create test image (28x28)
    image = np.random.rand(1, 28, 28).astype(np.float32)
    noise_level = 0.1  # ε = 0.1

    # Create engine and propagator
    engine = AffineExpressionEngine()
    propagator = BoundPropagator(engine)

    # Create input expressions
    input_exprs = engine.create_input_expressions(image, noise_level)

    # Propagate through network
    output_exprs = propagator.propagate(input_exprs, layers, image.shape)

    # Get bounds
    output_bounds = [expr.get_bounds() for expr in output_exprs]

    print(f"Output bounds computed for {len(output_bounds)} logits")
    print(f"Logit 0: [{output_bounds[0][0]:.4f}, {output_bounds[0][1]:.4f}]")

VGG16 Partial Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large networks, use partial evaluation:

.. code-block:: python

    from abstractnn.partial_evaluator import verify_partial_soundness
    import numpy as np

    # Create test image (224x224x3)
    image = np.random.rand(3, 224, 224).astype(np.float32)
    
    # Normalize for ImageNet
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = (image - mean) / std

    # Verify first 3 layers
    result = verify_partial_soundness(
        model_path='vgg16.onnx',
        image=image,
        noise_level=0.01,
        num_layers=3,
        num_mc_samples=100
    )

    if result['success']:
        report = result['soundness_report']
        print(f"Sound: {report['is_sound']}")
        print(f"Coverage: {report['coverage_ratio']*100:.2f}%")
    else:
        print(f"Error: {result['error']}")

Soundness Checking
------------------

Verify Your Bounds
~~~~~~~~~~~~~~~~~~

Compare formal bounds with Monte Carlo sampling:

.. code-block:: python

    from abstractnn.soundness_checker import SoundnessChecker
    import numpy as np

    checker = SoundnessChecker()

    # Your formal bounds
    formal_bounds = [(0.1, 0.9), (0.2, 0.8), (0.0, 1.0)]

    # Observed values from sampling
    observed_samples = np.array([
        [0.15, 0.22, 0.05],
        [0.85, 0.75, 0.95],
        [0.12, 0.25, 0.10]
    ])

    # Check soundness
    report = checker.check_soundness(
        formal_bounds,
        observed_samples,
        tolerance=1e-6
    )

    print(f"Is sound: {report['is_sound']}")
    print(f"Violations: {report['violation_count']}")
    
    if not report['is_sound']:
        for v in report['violations'][:5]:
            print(f"  Index {v['index']}: "
                  f"obs=[{v['observed_min']:.4f}, {v['observed_max']:.4f}] "
                  f"formal=[{v['formal_min']:.4f}, {v['formal_max']:.4f}]")

Common Workflows
----------------

Robustness Verification
~~~~~~~~~~~~~~~~~~~~~~~

Check if a prediction is robust:

.. code-block:: python

    def is_prediction_robust(model_path, image, epsilon, true_class):
        """
        Check if prediction remains stable under ε perturbation.
        
        Args:
            model_path: Path to ONNX model
            image: Input image
            epsilon: Perturbation radius
            true_class: True class label
            
        Returns:
            bool: True if robust
        """
        from modules.partial_evaluator import verify_partial_soundness
        
        result = verify_partial_soundness(
            model_path=model_path,
            image=image,
            noise_level=epsilon,
            num_layers=999,  # Full network
            num_mc_samples=100
        )
        
        if not result['success']:
            return False
        
        bounds = result['formal_bounds']
        
        # Check if true class has highest lower bound
        true_class_lower = bounds[true_class][0]
        
        for i, (lower, upper) in enumerate(bounds):
            if i != true_class and upper > true_class_lower:
                return False  # Not robust
        
        return True

Certified Accuracy
~~~~~~~~~~~~~~~~~~

Compute certified accuracy on a dataset:

.. code-block:: python

    def compute_certified_accuracy(model_path, test_images, test_labels, epsilon):
        """
        Compute certified accuracy.
        
        Args:
            model_path: Path to ONNX model
            test_images: Array of test images
            test_labels: True labels
            epsilon: Perturbation radius
            
        Returns:
            float: Certified accuracy
        """
        certified_correct = 0
        
        for image, label in zip(test_images, test_labels):
            if is_prediction_robust(model_path, image, epsilon, label):
                certified_correct += 1
        
        return certified_correct / len(test_images)

Configuration Tips
------------------



Performance Tuning
~~~~~~~~~~~~~~~~~~

Optimize for speed:

.. code-block:: python

    # Disable detailed reporting
    propagator = BoundPropagator(engine, enable_reporting=False)

    # Reduce Monte Carlo samples
    result = verify_partial_soundness(
        model_path=model_path,
        image=image,
        noise_level=epsilon,
        num_mc_samples=50  # Instead of 100
    )


Next Steps
----------

Now that you've completed the quick start:

1. **Tutorials**: Learn specific techniques in :doc:`tutorials/index`
2. **API Reference**: Explore the full API in :doc:`api/modules`
3. **Examples**: See complete examples in :doc:`user_guide/vgg16_example`
4. **Advanced**: Optimization techniques in :doc:`advanced/optimization`

Common Issues
-------------

See :doc:`advanced/troubleshooting` for solutions to common problems.

Need Help?
----------

- GitHub Issues: Report bugs and feature requests
- Documentation: Search this documentation
- Examples: Check the ``tests/`` directory for working examples