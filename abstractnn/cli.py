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


def evaluate():
    """Affine evaluation command"""
    parser = argparse.ArgumentParser(
        description="Formal evaluation using affine arithmetic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Evaluate with default FMNIST model
    abstractnn-eval

    # Evaluate custom model
    abstractnn-eval --model custom_model.onnx --image test.png --noise 0.1

    # With detailed report
    abstractnn-eval --detailed-report --export-report-csv report.csv
            """
        )
    
    parser.add_argument(
        "--model", "-m",
        default="examples/fmnist_cnn.onnx",
        help="Path to ONNX model file (default: examples/fmnist_cnn.onnx)"
    )
    
    parser.add_argument(
        "--image", "-i",
        default="examples/fmnist_sample_0_Ankle_boot.png",
        help="Path to input image (default: examples/fmnist_sample_0_Ankle_boot.png)"
    )
    
    parser.add_argument(
        "--noise", "-n",
        type=float,
        default=0.001,
        help="Noise level epsilon (default: 0.05)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="results.json",
        help="Output JSON file (default: results.json)"
    )
    
    parser.add_argument(
        "--activation-relaxation",
        default="linear",
        choices=["linear", "quadratic"],
        help="Activation relaxation type (default: linear)"
    )
    
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device to use (default: cpu)"
    )
    
    parser.add_argument(
        "--detailed-report",
        action="store_true",
        help="Generate detailed propagation report"
    )
    
    parser.add_argument(
        "--export-report-csv",
        type=str,
        metavar="FILE",
        help="Export detailed report as CSV"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"abstractNN {__version__}"
    )
    
    args = parser.parse_args()
    
    # Import affine_eval here to avoid circular imports
    try:
        # Try to import from parent directory
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        from affine_eval import evaluate_model, convert_to_json_serializable
    except ImportError:
        print("Error: Could not import affine_eval module", file=sys.stderr)
        sys.exit(1)
    
    # Check if files exist
    model_path = Path(args.model)
    image_path = Path(args.image)
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        print(f"Please ensure the file exists or provide a valid path with --model", file=sys.stderr)
        sys.exit(1)
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        print(f"Please ensure the file exists or provide a valid path with --image", file=sys.stderr)
        sys.exit(1)
    
    print(f"abstractNN Affine Evaluator v{__version__}")
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Noise: {args.noise}")
    print(f"Device: {args.device}")
    print()
    
    try:
        # Run evaluation
        results = evaluate_model(
            model_path=str(model_path),
            input_image=str(image_path),
            noise_level=args.noise,
            activation_relaxation=args.activation_relaxation,
            device=args.device,
            enable_detailed_report=args.detailed_report
        )
        
        # Convert to JSON serializable
        json_results = convert_to_json_serializable(results)
        
        # Save results
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Robust class: {results.get('robust_class', 'None')}")
        print(f"Execution time: {results['execution_time']:.2f}s")
        print(f"Results saved to: {output_path}")
        
        # Display bounds summary
        if 'bounds_per_class' in results and results['bounds_per_class']:
            print(f"\nBounds per class (sample):")
            for class_id, bounds in list(results['bounds_per_class'].items())[:5]:
                print(f"  Class {class_id}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
        
        # Export CSV if requested
        if args.export_report_csv and 'detailed_report' in results:
            csv_path = Path(args.export_report_csv)
            # Note: This would require implementing CSV export in affine_eval
            print(f"\n⚠️  CSV export not yet implemented")
            print(f"   Detailed report available in JSON: {output_path}")
        
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


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
""")


if __name__ == "__main__":
    verify()
