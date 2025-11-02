.

AbstractNN Documentation
=======================

**AbstractNN** is a Python library for **formal verification of neural networks** using abstract interpretation and affine arithmetic. It provides **mathematically sound** guarantees about network behavior under input perturbations.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   overview
   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/affine_expressions
   user_guide/propagation
   user_guide/relaxation
   user_guide/soundness
   user_guide/vgg16_example
   cli

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/abstractnn
   api/affine_engine
   api/bound_propagator
   api/relaxer
   api/onnx_parser
   api/partial_evaluator
   api/soundness_checker

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/optimization
   advanced/custom_layers
   advanced/troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/testing
   development/architecture

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`