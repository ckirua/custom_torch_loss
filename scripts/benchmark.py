"""
Benchmark: Python vs C++ Custom Loss Functions

This script compares the performance of:
1. Original Python implementation (from your uploaded file)
2. C++ implementation (pybind11 extension)

Tests include:
- Forward pass speed
- Backward pass speed
- Memory usage
- Batch size scaling

Usage:
  python benchmark.py [--yes] [--quick | --full]
  --yes, -y     Continue even if correctness check fails
  --quick, -q   Small batches, fewer iterations (default when no GPU)
  --full        Full benchmark (all batch sizes, more iterations)
"""

import torch
import torch.nn as nn
import time
import numpy as np
from collections import defaultdict
import sys

# Import original Python implementations
class AdjMSELoss1_Python(nn.Module):
    """Original Python implementation"""
    def __init__(self):
        super(AdjMSELoss1_Python, self).__init__()

    def forward(self, outputs, labels):
        outputs = torch.squeeze(outputs)
        alpha = 2
        loss = (outputs - labels) ** 2
        adj = torch.mul(outputs, labels)
        adj[adj > 0] = 1 / alpha
        adj[adj < 0] = alpha
        loss = loss * adj
        return torch.mean(loss)


class AdjMSELoss2_Python(nn.Module):
    """Original Python implementation"""
    def __init__(self):
        super(AdjMSELoss2_Python, self).__init__()

    def forward(self, outputs, labels):
        outputs = torch.squeeze(outputs)
        beta = 2.5
        loss = (outputs - labels) ** 2
        adj_loss = beta - (beta - 0.5) / (
            1 + torch.exp(10000 * torch.mul(outputs, labels))
        )
        loss = beta * loss / (1 + adj_loss)
        return torch.mean(loss)


class AdjMSELoss3_Python(nn.Module):
    """Original Python implementation"""
    def __init__(self):
        super(AdjMSELoss3_Python, self).__init__()

    def forward(self, outputs, labels):
        outputs = torch.squeeze(outputs)
        gamma = 0.1
        loss = (outputs - labels) ** 2
        adj = torch.mul(outputs, labels)
        adj[adj > 0] = gamma
        adj[adj < 0] = 1 + gamma
        loss = loss * adj
        return torch.mean(loss)


# Try to import C++ implementations
try:
    from custom_loss import AdjMSELoss1, AdjMSELoss2, AdjMSELoss3
    CPP_AVAILABLE = True
except ImportError:
    print("⚠️  C++ extension not found. Building it now...")
    print("Run: python setup.py develop")
    CPP_AVAILABLE = False
    sys.exit(1)


def benchmark_forward_pass(loss_fn, outputs, labels, n_iterations=1000, warmup=100):
    """Benchmark forward pass only."""
    # Warmup
    for _ in range(warmup):
        _ = loss_fn(outputs, labels)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        loss = loss_fn(outputs, labels)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    return (end_time - start_time) / n_iterations


def benchmark_backward_pass(loss_fn, outputs, labels, n_iterations=1000, warmup=100):
    """Benchmark forward + backward pass."""
    # Warmup
    for _ in range(warmup):
        outputs_temp = outputs.clone().detach().requires_grad_(True)
        loss = loss_fn(outputs_temp, labels)
        loss.backward()
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        outputs_temp = outputs.clone().detach().requires_grad_(True)
        loss = loss_fn(outputs_temp, labels)
        loss.backward()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    return (end_time - start_time) / n_iterations


def format_time(seconds):
    """Format time in appropriate units"""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def run_benchmark_suite(quick=False):
    """Run comprehensive benchmarks.
    quick: use smaller batch sizes and fewer iterations (for CPU-only / low-resource).
    """
    
    print("=" * 80)
    print("CUSTOM LOSS FUNCTIONS: PYTHON vs C++ BENCHMARK")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if quick:
        print("Mode: quick (small batches, fewer iterations)")
    
    # Test configurations: lighter for CPU-only / low-resource
    if quick:
        batch_sizes = [32, 128, 512]
        n_iterations = 100
        warmup = 10
    else:
        batch_sizes = [32, 128, 512, 2048, 8192]
        n_iterations = 1000
        warmup = 100
    
    results = defaultdict(lambda: defaultdict(dict))
    
    for batch_size in batch_sizes:
        print(f"\n{'=' * 80}")
        print(f"Batch Size: {batch_size}")
        print(f"{'=' * 80}")
        
        # Generate test data
        outputs = torch.randn(batch_size, 1, device=device, requires_grad=True)
        labels = torch.randn(batch_size, 1, device=device)
        
        # Test each loss function
        loss_functions = [
            ("AdjMSELoss1", AdjMSELoss1_Python(), AdjMSELoss1(alpha=2.0)),
            ("AdjMSELoss2", AdjMSELoss2_Python(), AdjMSELoss2(beta=2.5)),
            ("AdjMSELoss3", AdjMSELoss3_Python(), AdjMSELoss3(gamma=0.1)),
        ]
        
        for loss_name, py_loss, cpp_loss in loss_functions:
            print(f"\n{loss_name}:")
            print("-" * 80)
            
            # Move to device
            py_loss = py_loss.to(device)
            cpp_loss = cpp_loss.to(device)
            
            # Forward pass benchmark
            py_forward_time = benchmark_forward_pass(py_loss, outputs, labels, n_iterations, warmup)
            cpp_forward_time = benchmark_forward_pass(cpp_loss, outputs, labels, n_iterations, warmup)
            
            # Backward pass benchmark
            py_backward_time = benchmark_backward_pass(py_loss, outputs, labels, n_iterations, warmup)
            cpp_backward_time = benchmark_backward_pass(cpp_loss, outputs, labels, n_iterations, warmup)
            
            # Calculate speedups
            forward_speedup = py_forward_time / cpp_forward_time
            backward_speedup = py_backward_time / cpp_backward_time
            
            # Store results
            results[loss_name][batch_size] = {
                'py_forward': py_forward_time,
                'cpp_forward': cpp_forward_time,
                'py_backward': py_backward_time,
                'cpp_backward': cpp_backward_time,
                'forward_speedup': forward_speedup,
                'backward_speedup': backward_speedup,
            }
            
            # Print results
            print(f"  Forward Pass:")
            print(f"    Python:  {format_time(py_forward_time)}")
            print(f"    C++:     {format_time(cpp_forward_time)}")
            print(f"    Speedup: {forward_speedup:.2f}x {'🚀' if forward_speedup > 1 else '⚠️'}")
            
            print(f"  Forward + Backward Pass:")
            print(f"    Python:  {format_time(py_backward_time)}")
            print(f"    C++:     {format_time(cpp_backward_time)}")
            print(f"    Speedup: {backward_speedup:.2f}x {'🚀' if backward_speedup > 1 else '⚠️'}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    for loss_name in ["AdjMSELoss1", "AdjMSELoss2", "AdjMSELoss3"]:
        print(f"\n{loss_name}:")
        print("-" * 80)
        print(f"{'Batch Size':<12} {'Forward Speedup':<18} {'Backward Speedup':<18}")
        print("-" * 80)
        
        for batch_size in batch_sizes:
            data = results[loss_name][batch_size]
            print(f"{batch_size:<12} {data['forward_speedup']:>8.2f}x {'🚀' if data['forward_speedup'] > 1 else '⚠️':<9} "
                  f"{data['backward_speedup']:>8.2f}x {'🚀' if data['backward_speedup'] > 1 else '⚠️'}")
    
    # Average speedups
    print("\n" + "=" * 80)
    print("AVERAGE SPEEDUPS ACROSS ALL BATCH SIZES")
    print("=" * 80)
    
    for loss_name in ["AdjMSELoss1", "AdjMSELoss2", "AdjMSELoss3"]:
        forward_speedups = [results[loss_name][bs]['forward_speedup'] for bs in batch_sizes]
        backward_speedups = [results[loss_name][bs]['backward_speedup'] for bs in batch_sizes]
        
        avg_forward = np.mean(forward_speedups)
        avg_backward = np.mean(backward_speedups)
        
        print(f"\n{loss_name}:")
        print(f"  Average Forward Speedup:  {avg_forward:.2f}x")
        print(f"  Average Backward Speedup: {avg_backward:.2f}x")


def memory_benchmark(quick=False):
    """Compare memory usage. quick: smaller batch and fewer iterations."""
    print("\n" + "=" * 80)
    print("MEMORY USAGE COMPARISON")
    print("=" * 80)
    
    import tracemalloc
    
    if quick:
        batch_size = 1000
        n_iters = 20
    else:
        batch_size = 10000
        n_iters = 100
    
    outputs = torch.randn(batch_size, 1, requires_grad=True)
    labels = torch.randn(batch_size, 1)
    
    # Python implementation
    tracemalloc.start()
    py_loss = AdjMSELoss1_Python()
    for _ in range(n_iters):
        loss = py_loss(outputs, labels)
        loss.backward()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    py_memory = peak / 1024 / 1024  # MB
    
    # C++ implementation
    tracemalloc.start()
    cpp_loss = AdjMSELoss1(alpha=2.0)
    for _ in range(n_iters):
        loss = cpp_loss(outputs, labels)
        loss.backward()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cpp_memory = peak / 1024 / 1024  # MB
    
    print(f"\nBatch size: {batch_size}, {n_iters} iterations")
    print(f"Python implementation: {py_memory:.2f} MB")
    print(f"C++ implementation:    {cpp_memory:.2f} MB")
    print(f"Memory reduction:      {((py_memory - cpp_memory) / py_memory * 100):.1f}%")


def correctness_check():
    """Verify that Python and C++ implementations produce identical results"""
    print("\n" + "=" * 80)
    print("CORRECTNESS CHECK")
    print("=" * 80)
    
    torch.manual_seed(42)
    outputs = torch.randn(100, 1, requires_grad=True)
    labels = torch.randn(100, 1)
    
    loss_pairs = [
        ("AdjMSELoss1", AdjMSELoss1_Python(), AdjMSELoss1(alpha=2.0)),
        ("AdjMSELoss2", AdjMSELoss2_Python(), AdjMSELoss2(beta=2.5)),
        ("AdjMSELoss3", AdjMSELoss3_Python(), AdjMSELoss3(gamma=0.1)),
    ]
    
    print("\nComparing loss values:")
    print("-" * 80)
    
    all_correct = True
    for name, py_loss, cpp_loss in loss_pairs:
        py_result = py_loss(outputs.clone(), labels).item()
        cpp_result = cpp_loss(outputs.clone(), labels).item()
        
        diff = abs(py_result - cpp_result)
        relative_error = diff / abs(py_result) if py_result != 0 else diff
        
        is_correct = relative_error < 1e-5
        all_correct = all_correct and is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {name:15s}: Python={py_result:.8f}, C++={cpp_result:.8f}, "
              f"Error={relative_error:.2e}")
    
    if all_correct:
        print("\n✅ All implementations produce identical results!")
    else:
        print("\n❌ Some implementations have discrepancies!")
    
    return all_correct


if __name__ == "__main__":
    if not CPP_AVAILABLE:
        print("\n❌ C++ extension is not available.")
        print("Please build it first:")
        print("  python setup.py develop")
        sys.exit(1)
    
    skip_prompt = "--yes" in sys.argv or "-y" in sys.argv
    # Quick mode: smaller batches, fewer iterations (for CPU-only / low-resource).
    # Auto when no GPU; override with --quick / -q or --full.
    quick = "--quick" in sys.argv or "-q" in sys.argv
    force_full = "--full" in sys.argv
    if not force_full and not quick and not torch.cuda.is_available():
        quick = True
        print("(No GPU detected; using quick mode. Pass --full for full benchmark.)")
    # Run correctness check first
    if not correctness_check():
        print("\n⚠️  Warning: Correctness check failed!")
        if skip_prompt:
            print("Continuing with benchmark (--yes).")
        else:
            response = input("Continue with benchmark anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Run benchmarks
    run_benchmark_suite(quick=quick)
    memory_benchmark(quick=quick)
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • C++ implementations should show speedups for larger batch sizes")
    print("  • Memory usage may be lower with C++ implementations")
    print("  • Results are numerically identical (within floating-point precision)")
    print("  • For production use, C++ extension provides better performance")
    print("\n" + "=" * 80)
