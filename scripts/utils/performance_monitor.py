import functools
import gc
import inspect
import threading
import time
import tracemalloc
from typing import Callable, Any

import psutil


def performance_monitor(func: Callable) -> Callable:
	"""
    A decorator that monitors and reports comprehensive function performance metrics.

    Metrics included:
    - Execution time: Total time taken for function execution in seconds
    - CPU usage: System-wide CPU utilization across all cores during function execution,
      measured as percentage of total available CPU capacity
    - Memory usage: Current Resident Set Size (RSS) - amount of process memory
      currently held in RAM, excluding swapped out memory
    - Peak memory: Highest point of memory allocation during execution, including
      temporary allocations that may have been freed
    - Thread utilization: Ratio of active threads to total available threads
    - I/O operations: Amount of data read from and written to disk in KB
    - Context switches:
        * Voluntary: Thread willingly gave up CPU
        * Involuntary: OS forced thread to yield CPU
    - Page faults: Number of times data had to be retrieved from disk due to not being in memory
    - Network usage: Amount of data sent and received over network in KB
    - Garbage collection: Number of garbage collection runs per generation (0/1/2)
    - Stack depth: Current depth of the call stack when function executes

    Args:
        func: The function to be monitored

    Returns:
        Wrapper function that includes performance monitoring
    """

	@functools.wraps(func)
	def wrapper(*args, **kwargs) -> Any:
		process = psutil.Process()

		# Collect initial metrics
		initial_metrics = {
			'cpu_time': process.cpu_percent(),
			'memory': process.memory_info().rss / 1024 / 1024  # MB
		}

		# Try to collect additional metrics that might not be available
		try:
			initial_metrics.update({
				'io': process.io_counters(),
				'ctx_switches': process.num_ctx_switches(),
				'page_faults': process.memory_full_info().num_page_faults,
				'net_io': psutil.net_io_counters(),
				'gc_count': gc.get_count()
			})
		except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
			print(f"Warning: Some metrics may be unavailable - {str(e)}")

		# Start monitoring
		tracemalloc.start()
		start_time = time.perf_counter()

		# Execute the function
		result = func(*args, **kwargs)

		# Collect final metrics
		end_time = time.perf_counter()
		final_cpu_time = process.cpu_percent()
		final_memory = process.memory_info().rss / 1024 / 1024
		current_mem, peak_mem = tracemalloc.get_traced_memory()
		tracemalloc.stop()

		# Calculate thread metrics
		thread_count = process.num_threads()
		thread_utilization = len(threading.enumerate()) / thread_count

		# Calculate differential metrics if initial values were available
		metrics = {
			'execution_time': end_time - start_time,
			'memory_used': final_memory - initial_metrics['memory'],
			'cpu_usage': final_cpu_time - initial_metrics['cpu_time'],
			'peak_memory': peak_mem / 1024 / 1024,  # Convert to MB
			'thread_count': thread_count,
			'thread_utilization': thread_utilization,
			'stack_depth': len(inspect.stack())
		}

		if 'io' in initial_metrics:
			final_io = process.io_counters()
			metrics.update({
				'read_kb': (final_io.read_bytes - initial_metrics[
					'io'].read_bytes) / 1024,
				'write_kb': (final_io.write_bytes - initial_metrics[
					'io'].write_bytes) / 1024
			})

		if 'ctx_switches' in initial_metrics:
			final_ctx = process.num_ctx_switches()
			metrics.update({
				'voluntary_ctx': final_ctx.voluntary - initial_metrics[
					'ctx_switches'].voluntary,
				'involuntary_ctx': final_ctx.involuntary - initial_metrics[
					'ctx_switches'].involuntary
			})

		if 'page_faults' in initial_metrics:
			metrics['page_faults'] = (
				process.memory_full_info().num_page_faults - initial_metrics[
				'page_faults']
			)

		if 'net_io' in initial_metrics:
			final_net = psutil.net_io_counters()
			metrics.update({
				'bytes_sent_kb': (final_net.bytes_sent - initial_metrics[
					'net_io'].bytes_sent) / 1024,
				'bytes_received_kb': (final_net.bytes_recv - initial_metrics[
					'net_io'].bytes_recv) / 1024
			})

		if 'gc_count' in initial_metrics:
			final_gc_count = gc.get_count()
			metrics['gc_runs'] = tuple(
				f - i for f, i in
				zip(final_gc_count, initial_metrics['gc_count'])
			)

		# Print performance report
		print(f"\nPerformance Metrics for {func.__name__}:")
		print("=" * 70)

		# Basic metrics
		print(f"Execution Time: {metrics['execution_time']:.4f} seconds")
		print(f"CPU Usage: {metrics['cpu_usage']:.2f}%")
		print(f"Memory Usage: {metrics['memory_used']:.2f} MB")
		print(f"Peak Memory: {metrics['peak_memory']:.2f} MB")
		print(f"Thread Count: {metrics['thread_count']}")
		print(f"Thread Utilization: {metrics['thread_utilization']:.2%}")

		# Optional metrics
		if 'read_kb' in metrics:
			print(
				f"I/O Operations - Read: {metrics['read_kb']:.2f} KB, Write: {metrics['write_kb']:.2f} KB")

		if 'voluntary_ctx' in metrics:
			print(f"Context Switches - Voluntary: {metrics['voluntary_ctx']}, "
				  f"Involuntary: {metrics['involuntary_ctx']}")

		if 'page_faults' in metrics:
			print(f"Page Faults: {metrics['page_faults']}")

		if 'bytes_sent_kb' in metrics:
			print(f"Network Usage - Sent: {metrics['bytes_sent_kb']:.2f} KB, "
				  f"Received: {metrics['bytes_received_kb']:.2f} KB")

		if 'gc_runs' in metrics:
			print(f"GC Runs (Gen 0/1/2): {metrics['gc_runs']}")

		print(f"Stack Depth: {metrics['stack_depth']}")
		print("=" * 70 + "\n")

		return result

	return wrapper