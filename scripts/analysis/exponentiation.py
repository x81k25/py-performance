# Import necessary libraries for performance monitoring, parallel processing, and numerical operations
import gc  # Garbage collector for memory management
import multiprocessing  # For parallel processing capabilities
import sys  # System-specific parameters and functions
from typing import Any, Callable  # Type hints for better code documentation

# Scientific and numerical computing libraries
import cupy as cp  # GPU accelerated array operations
import numpy as np  # CPU-based numerical operations
import pandas as pd  # Data manipulation and analysis
from joblib import Parallel, delayed  # Tools for parallel processing

# Custom performance monitoring decorator
from scripts.utils.performance_monitor import performance_monitor

# Define constants for data size and parallel processing
SIZE = int(1e8)  # Creating a large dataset (100 million elements) to stress test performance 
MAX_THREADS = multiprocessing.cpu_count()  # Get number of CPU cores for parallel processing

# Base exponentiation function with performance monitoring
# This serves as our baseline for comparing different implementation methods
@performance_monitor
def exponentiate(base, power):
   return base ** power

# List comprehension based exponentiation
# This demonstrates the performance characteristics of Python's native list operations
@performance_monitor 
def for_loop_exponentiate(list_vec: list, power: int):
   return [x ** power for x in list_vec]

# NumPy-optimized power function
# Leverages NumPy's vectorized operations for improved performance over native Python
@performance_monitor
def exponentiate_np_power(base, power):
   return np.power(base, power)

# Helper function for parallel processing
# Processes a chunk of the array in parallel to distribute computational load
def exp_chunk(chunk, power):
   return np.power(chunk, power)

# Parallel processing implementation using joblib
# Splits the workload across multiple CPU cores for potential performance gains
@performance_monitor
def parallel_exponentiate(array, power, n_jobs=MAX_THREADS):
   chunks = np.array_split(array, n_jobs)  # Split array into chunks for parallel processing
   results = Parallel(n_jobs=n_jobs)(
       delayed(exp_chunk)(chunk, power) for chunk in chunks
   )
   return

# GPU-accelerated implementation using CuPy
# Offloads computation to GPU for potentially massive performance gains
@performance_monitor
def cupy_exponential(arr, power):
   gpu_arr = cp.asarray(arr)  # Transfer data to GPU memory
   result = cp.power(gpu_arr, power)  # Perform computation on GPU
   return cp.asnumpy(result)  # Transfer results back to CPU memory

# Data Generation and Performance Testing Section

# Create test data in different formats to compare performance across implementations
list_vec = np.random.randint(1, 101, size=SIZE, dtype=np.int64)  # Raw integer array
numpy_arr = np.array(list_vec)  # NumPy array for vectorized operations
pandas_ser = pd.Series(list_vec)  # Pandas Series for comparison

# Memory usage monitoring
# Important for understanding space complexity of different implementations
bytes_size = sys.getsizeof(list_vec)
megabytes = bytes_size / (1024 * 1024)
print(f"Memory usage: {megabytes:.2f} MB")
print(f"Logical cores: {MAX_THREADS}")

# Performance Comparison Testing
# Running each implementation to compare their performance characteristics

# Test 1: Basic NumPy array exponentiation
exponentiate(numpy_arr, 100)

# Test 2: Pandas Series exponentiation
exponentiate(pandas_ser, 100)

# Test 3: NumPy optimized power function
performance_monitor(exponentiate_np_power)(numpy_arr, 100)

# Test 4: Parallel processing implementation
parallel_exponentiate(numpy_arr, 100)

# Test 5: GPU-accelerated implementation
cupy_exponential(numpy_arr, 100)

# This script demonstrates a comprehensive performance comparison between:
# 1. Native Python operations
# 2. NumPy vectorized operations
# 3. Parallel CPU processing
# 4. GPU acceleration
# The goal is to understand the performance characteristics and trade-offs
# of different implementation approaches for large-scale numerical computations