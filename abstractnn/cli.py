"""
Command Line Interface for abstractNN
======================================

Provides command-line tools for neural network verification.
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

from . import __version__, __author__
from .partial_evaluator import verify_partial_soundness


def verify():
    """Main verification command"""
    parser = argparse.ArgumentParser(
        description="Verify neural network robustness using abstractNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify MNIST model
  abstractnn-verify --model mnist.onnx --image test.npy --epsilon 0.1

  # Verify VGG16 first 5 layers
  abstractnn-verify --model vgg16.onnx --image img.npy --epsilon 0.01 --layers 5

  # Save detailed report
  abstractnn-verify --model model.onnx --image data.npy --epsilon 0.05 --output report.json
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to ONNX model file"
    )
    
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to input image (.npy or .png)"
    )
    
    parser.add_argument(
        "--epsilon", "-e",
        type=float,
        default=0.01,
        help="Perturbation radius (L-infinity norm)"
    )
    
    parser.add_argument(
        "--layers", "-l",
        type=int,
        default=999,
        help="Number of layers to verify (default: all)"
    )
    
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=100,
        help="Monte Carlo samples for soundness checking"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output JSON report file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"abstractNN {__version__}"
    )
    
    args = parser.parse_args()
    
    # Load image
    image_path = Path(args.image)
    if image_path.suffix == ".npy":
        image = np.load(args.image)
    else:
        print(f"Error: Only .npy format supported currently", file=sys.stderr)
        sys.exit(1)
    
    if not args.verbose:
        # Suppress detailed output
        import warnings
        warnings.filterwarnings("ignore")
    
    print(f"abstractNN v{__version__}")
    print(f"Verifying: {args.model}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Layers: {args.layers if args.layers < 999 else 'all'}")
    print()
    
    # Run verification
    result = verify_partial_soundness(
        model_path=args.model,
        image=image,
        noise_level=args.epsilon,
        num_layers=args.layers,
        num_mc_samples=args.samples
    )
    
    # Display results
    if result['success']:
        report = result['soundness_report']
        print("=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        print(f"Sound: {'✅ YES' if report['is_sound'] else '❌ NO'}")
        print(f"Coverage: {report['coverage_ratio']*100:.1f}%")
        print(f"Violations: {report['violation_count']}/{report['total_outputs']}")
        print()
        
        if not report['is_sound'] and report['violations']:
            print("Top violations:")
            for i, v in enumerate(report['violations'][:5], 1):
                print(f"  {i}. Index {v['index']}: "
                      f"[{v['formal_min']:.4f}, {v['formal_max']:.4f}] "
                      f"vs observed [{v['observed_min']:.4f}, {v['observed_max']:.4f}]")
        
        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nFull report saved to: {args.output}")
        
        sys.exit(0 if report['is_sound'] else 1)
    else:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(2)


def info():
    """Display library information"""
    print(f"""
abstractNN - Neural Network Formal Verification
================================================

Version: {__version__}
Author: {__author__}
License: MIT

Installation:
  pip install abstractNN

Documentation:
  https://abstractnn.readthedocs.io

GitHub:
  https://github.com/flyworthi/abstractNN

Quick Start:
  from abstractnn import AffineEngine, BoundPropagator, ONNXParser
  
  parser = ONNXParser('model.onnx')
  layers = parser.parse()
  
  engine = AffineEngine()
  propagator = BoundPropagator(engine)
  
  # Verify network...

For help:
  abstractnn-verify --help
""")


if __name__ == "__main__":
    verify()
