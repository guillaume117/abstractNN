

Command Line Interface
======================

abstractNN provides three command-line tools for neural network verification and evaluation.

Overview
--------

.. code-block:: bash

   # Soundness verification
   abstractnn-verify --model model.onnx --image test.npy --epsilon 0.01

   # Formal evaluation
   abstractnn-eval --model model.onnx --image test.png --noise 0.05

   # Library information
   abstractnn-info

abstractnn-verify
-----------------

Verify neural network soundness by comparing formal bounds with Monte Carlo sampling.

Synopsis
~~~~~~~~

.. code-block:: text

   abstractnn-verify [OPTIONS]

Description
~~~~~~~~~~~

This command performs soundness verification by:

1. Computing formal bounds using affine arithmetic
2. Running Monte Carlo sampling with random perturbations
3. Comparing formal bounds with observed values
4. Reporting violations and coverage metrics

Options
~~~~~~~

.. option:: -m, --model MODEL

   **Required.** Path to ONNX model file.

.. option:: -i, --image IMAGE

   **Required.** Path to input image (.npy format).

.. option:: -e, --epsilon FLOAT

   Perturbation radius (L-infinity norm). Default: 0.01

.. option:: -l, --layers INT

   Number of layers to verify. Default: all layers (999)

.. option:: -s, --samples INT

   Monte Carlo samples for soundness checking. Default: 100

.. option:: -o, --output FILE

   Output JSON report file. Optional.

.. option:: -v, --verbose

   Enable verbose output.

.. option:: --version

   Show version and exit.

Examples
~~~~~~~~

Basic verification:

.. code-block:: bash

   abstractnn-verify --model mnist.onnx --image test.npy --epsilon 0.1

Verify first 5 layers of VGG16:

.. code-block:: bash

   abstractnn-verify --model vgg16.onnx --image img.npy --epsilon 0.01 --layers 5

With detailed report:

.. code-block:: bash

   abstractnn-verify \
       --model model.onnx \
       --image data.npy \
       --epsilon 0.05 \
       --output report.json \
       --verbose

Output Format
~~~~~~~~~~~~~

The command outputs verification results to stdout:

.. code-block:: text

   ======================================================
   VERIFICATION RESULTS
   ======================================================
   Sound: ✅ YES
   Coverage: 98.5%
   Violations: 15/1000

If ``--output`` is specified, a detailed JSON report is saved:

.. code-block:: json

   {
     "success": true,
     "soundness_report": {
       "is_sound": true,
       "violation_count": 0,
       "total_outputs": 1000,
       "coverage_ratio": 1.0,
       "violations": [],
       "safety_margins_lower": [...],
       "safety_margins_upper": [...]
     },
     "num_layers": 5,
     "num_outputs": 1000,
     "noise_level": 0.01,
     "num_mc_samples": 100
   }

Exit Codes
~~~~~~~~~~

- **0**: Verification successful and sound
- **1**: Verification successful but not sound (violations detected)
- **2**: Error during verification

abstractnn-eval
---------------

Perform formal evaluation using affine arithmetic to compute guaranteed output bounds.

Synopsis
~~~~~~~~

.. code-block:: text

   abstractnn-eval [OPTIONS]

Description
~~~~~~~~~~~

This command performs formal evaluation by:

1. Parsing the ONNX model
2. Creating symbolic affine expressions for inputs
3. Propagating expressions through network layers
4. Computing guaranteed bounds for all output classes
5. Identifying robustly classified inputs

Options
~~~~~~~

.. option:: -m, --model MODEL

   Path to ONNX model file. Default: ``examples/fmnist_cnn.onnx``

.. option:: -i, --image IMAGE

   Path to input image (.png or .npy). Default: ``examples/fmnist_sample_0_Ankle_boot.png``

.. option:: -n, --epsilon FLOAT

   Noise level epsilon. Default: 0.001

.. option:: -o, --output FILE

   Output JSON file. Default: ``results.json``


.. option:: --device {cpu,gpu}

   Device to use. Default: ``cpu``

.. option:: --detailed-report

   Generate detailed propagation report.

.. option:: --export-report-csv FILE

   Export detailed report as CSV.

.. option:: --version

   Show version and exit.

Examples
~~~~~~~~

Use default FMNIST example:

.. code-block:: bash

   abstractnn-eval

Custom model and image:

.. code-block:: bash

   abstractnn-eval \
       --model custom_model.onnx \
       --image test.png \
       --epsilon 0.01

With detailed report:

.. code-block:: bash

   abstractnn-eval \
       --model mymodel.onnx \
       --image myimage.png \
       --epsilon 0.05 \
       --detailed-report \
       --output results.json

Output Format
~~~~~~~~~~~~~

The command outputs evaluation results to stdout:

.. code-block:: text

   ==========================================================
   EVALUATION RESULTS
   ==========================================================
   Robust class: 5
   Execution time: 12.34s
   Results saved to: results.json

   Bounds per class (sample):
     Class 0: [-2.1543, 1.8765]
     Class 1: [-1.2345, 2.3456]
     Class 2: [-0.9876, 3.4567]
     Class 3: [-1.5432, 2.1098]
     Class 4: [-2.3456, 1.6543]

The JSON output contains:

.. code-block:: json

   {
     "bounds_per_class": {
       "0": [-2.1543, 1.8765],
       "1": [-1.2345, 2.3456],
       ...
     },
     "robust_class": 5,
     "intermediate_bounds": [...],
     "execution_time": 12.34,
     "noise_level": 0.05,
     "activation_relaxation": "linear"
   }

Activation Relaxation
~~~~~~~~~~~~~~~~~~~~~

:

**Linear Relaxation** (default):

For ReLU when :math:`l < 0 < u`:

.. math::

   \text{ReLU}(x) \in \left[0, u \cdot \frac{x - l}{u - l}\right]



abstractnn-info
---------------

Display library information and usage help.

Synopsis
~~~~~~~~

.. code-block:: text

   abstractnn-info

Description
~~~~~~~~~~~

Displays:

- Library version
- Author information
- Installation instructions
- Quick start guide
- Available commands
- Documentation links

Example Output
~~~~~~~~~~~~~~

.. code-block:: text

   abstractNN - Neural Network Formal Verification
   ================================================

   Version: 0.1.0
   Author: Guillaume Berthelot
   License: MIT

   Installation:
     pip install abstractNN

   Documentation:
     https://abstractnn.readthedocs.io

   GitHub:
     https://github.com/guillaume117/abstractNN

   Quick Start:
     from abstractnn import AffineEngine, BoundPropagator, ONNXParser
     
     parser = ONNXParser('model.onnx')
     layers = parser.parse()
     
     engine = AffineEngine()
     propagator = BoundPropagator(engine)
     
     # Verify network...

   Available Commands:
     abstractnn-verify  - Verify neural network soundness
     abstractnn-eval    - Formal evaluation with affine arithmetic
     abstractnn-info    - Display library information

   For help:
     abstractnn-verify --help
     abstractnn-eval --help

Common Workflows
----------------

Workflow 1: Quick Soundness Check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Verify with small epsilon
   abstractnn-verify --model model.onnx --image test.npy --epsilon 0.001

   # 2. If sound, increase epsilon
   abstractnn-verify --model model.onnx --image test.npy --epsilon 0.01

   # 3. Find maximum robust epsilon
   for eps in 0.001 0.005 0.01 0.05 0.1; do
       echo "Testing epsilon=$eps"
       abstractnn-verify --model model.onnx --image test.npy --epsilon $eps
   done

Workflow 2: Formal Evaluation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Evaluate with default settings
   abstractnn-eval --model mymodel.onnx --image test.png

   # 2. Generate detailed report
   abstractnn-eval \
       --model mymodel.onnx \
       --image test.png \
       --detailed-report \
       --output detailed_results.json

   # 3. Try different relaxation strategies
   abstractnn-eval \
       --model mymodel.onnx \
       --image test.png \
       --activation-relaxation quadratic

Workflow 3: Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   #!/bin/bash
   # Verify multiple images

   for img in images/*.npy; do
       echo "Verifying $img"
       abstractnn-verify \
           --model model.onnx \
           --image "$img" \
           --epsilon 0.01 \
           --output "results/$(basename $img .npy).json"
   done

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Error: "Model file not found"**

Ensure the model file exists:

.. code-block:: bash

   ls -lh model.onnx

**Error: "Only .npy format supported"**

For ``abstractnn-verify``, convert your image to NumPy format:

.. code-block:: python

   import numpy as np
   from PIL import Image
   
   img = Image.open('test.png')
   img_array = np.array(img, dtype=np.float32) / 255.0
   np.save('test.npy', img_array)

**Out of Memory**

Reduce the number of layers:

.. code-block:: bash

   abstractnn-verify --model large_model.onnx --image test.npy --layers 5

Or reduce Monte Carlo samples:

.. code-block:: bash

   abstractnn-verify --model model.onnx --image test.npy --samples 50

**Slow Execution**

Enable GPU (if available):

.. code-block:: bash

   abstractnn-eval --model model.onnx --image test.png --device gpu

Use simpler relaxation:

.. code-block:: bash

   abstractnn-eval --model model.onnx --image test.png --activation-relaxation linear

Performance Tips
~~~~~~~~~~~~~~~~

1. **Start small**: Test on a few layers first
2. **Use appropriate epsilon**: Smaller epsilon = faster
3. **Reduce samples**: For quick tests, use ``--samples 10``
4. **Cache results**: Save output with ``--output`` to avoid recomputation

See Also
--------

- :doc:`installation`: Installation guide
- :doc:`quickstart`: Quick start tutorial
- :doc:`api/modules`: API reference
- :doc:`advanced/troubleshooting`: Detailed troubleshooting