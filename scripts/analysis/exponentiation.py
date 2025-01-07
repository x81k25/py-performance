import cupy as cp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from matplotlib.gridspec import GridSpec
from scripts.utils.performance_monitor import performance_monitor

# Define constants for data size and parallel processing
SIZE = int(1e6)  # Size of our test data - 100 million elements
MAX_THREADS = multiprocessing.cpu_count()  # Maximum number of CPU threads available


@performance_monitor
def exponentiate(base, power):
	"""General exponentiation function that works with different data structures."""
	return base ** power


@performance_monitor
def for_loop_exponentiate(list_vec: list, power: int):
	"""Exponentiates elements using Python's native list comprehension."""
	return [x ** power for x in list_vec]


@performance_monitor
def exponentiate_np_power(base, power):
	"""Exponentiates using NumPy's optimized power function."""
	return np.power(base, power)


def exp_chunk(chunk, power):
	"""Helper function for parallel processing - operates on a chunk of data."""
	return np.power(chunk, power)


@performance_monitor
def parallel_exponentiate(array, power, n_jobs=MAX_THREADS):
	"""Exponentiates using parallel processing across multiple CPU cores."""
	chunks = np.array_split(array, n_jobs)
	results = Parallel(n_jobs=n_jobs)(
		delayed(exp_chunk)(chunk, power) for chunk in chunks
	)
	return np.concatenate(results)


@performance_monitor
def cupy_exponential(arr, power):
	"""Exponentiates using GPU acceleration via CuPy."""
	gpu_arr = cp.asarray(arr)
	result = cp.power(gpu_arr, power)
	return cp.asnumpy(result)


def run_multiple_iterations(n_iterations=5):
	"""
	Runs multiple iterations of each exponentiation method to gather statistical data.
	Returns a DataFrame containing performance metrics for each iteration.
	"""
	results_list = []

	# Define descriptions for each method
	descriptions = [
		"Uses Python's native list comprehension to perform exponentiation on each element of a Python list sequentially.",
		"Applies NumPy's basic array operations to perform vectorized exponentiation on a NumPy array.",
		"Leverages Pandas Series operations to perform vectorized exponentiation using underlying NumPy functionality.",
		"Utilizes NumPy's optimized power function for efficient vectorized exponentiation operations.",
		f"Splits the computation across {MAX_THREADS} CPU cores using parallel processing via joblib.",
		"Accelerates computation using GPU processing through CuPy's implementation of array operations."
	]

	short_descriptions = [
		"Python List Operation",
		"Basic NumPy Array",
		"Pandas Series Math",
		"NumPy Power Function",
		"CPU Parallel Processing",
		"GPU Accelerated Computation"
	]

	for i in range(n_iterations):
		# Create fresh data for each iteration
		list_vec = [np.random.randint(1, 101) for _ in range(SIZE)]
		numpy_arr = np.array(list_vec, dtype=np.int64)
		pandas_ser = pd.Series(numpy_arr)

		# Run all tests and collect metrics
		_, m1 = for_loop_exponentiate(list_vec, 100)
		_, m2 = exponentiate(numpy_arr, 100)
		_, m3 = exponentiate(pandas_ser, 100)
		_, m4 = exponentiate_np_power(numpy_arr, 100)
		_, m5 = parallel_exponentiate(numpy_arr, 100)
		_, m6 = cupy_exponential(numpy_arr, 100)

		iteration_metrics = [m1, m2, m3, m4, m5, m6]
		for j, metrics in enumerate(iteration_metrics):
			metrics['Method'] = short_descriptions[j]
			metrics['Description'] = descriptions[j]
			metrics['Iteration'] = i
			results_list.append(metrics)

	return pd.DataFrame(results_list)


# Run the performance tests multiple times
results_df = run_multiple_iterations()

# Calculate average metrics for each method
avg_results = results_df.groupby('Method').agg({
	'execution_time': 'mean',
	'memory_used': 'mean',
	'Description': 'first'  # Keep the description
}).reset_index()

# Set up the plotting style using seaborn's native styling system
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)

# Create figure with constrained_layout for better automatic spacing
fig = plt.figure(figsize=(15, 10), constrained_layout=True)
gs = GridSpec(2, 1, height_ratios=[1, 1], figure=fig)

# First subplot: Average Execution Time Comparison
ax1 = fig.add_subplot(gs[0])
time_plot = sns.barplot(
	data=avg_results,
	y='Method',
	x='execution_time',
	ax=ax1,
	hue='Method',
	legend=False
)
ax1.set_title('Average Execution Time Comparison Across Methods', pad=20)
ax1.set_xlabel('Execution Time (seconds)')
ax1.set_ylabel('Method')

# Add value labels on the bars
for i, v in enumerate(avg_results['execution_time']):
	ax1.text(v, i, f' {v:.3f}s', va='center')

# Second subplot: Average Memory Usage Comparison
ax2 = fig.add_subplot(gs[1])
memory_plot = sns.barplot(
	data=avg_results,
	y='Method',
	x='memory_used',
	ax=ax2,
	hue='Method',
	legend=False
)
ax2.set_title('Average Memory Usage Comparison Across Methods', pad=20)
ax2.set_xlabel('Memory Used (MB)')
ax2.set_ylabel('Method')

# Add value labels on the bars
for i, v in enumerate(avg_results['memory_used']):
	ax2.text(v, i, f' {v:.2f}MB', va='center')

# Add overall title
plt.suptitle(
	f'Performance Comparison of Different Exponentiation Methods\n(Array Size: {SIZE:,}, Averaged over {len(results_df.Iteration.unique())} iterations)',
	fontsize=14,
	y=1.02
)

# Save the comparison plot
plt.savefig('exponentiation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create statistical distribution plot with modern styling
plt.figure(figsize=(15, 10))

# Create violin plot for execution time distribution
ax1 = plt.subplot(211)
sns.violinplot(data=results_df, x='execution_time', y='Method', ax=ax1)
ax1.set_title('Distribution of Execution Times Across Methods')
ax1.set_xlabel('Execution Time (seconds)')

# Create violin plot for memory usage distribution
ax2 = plt.subplot(212)
sns.violinplot(data=results_df, x='memory_used', y='Method', ax=ax2)
ax2.set_title('Distribution of Memory Usage Across Methods')
ax2.set_xlabel('Memory Used (MB)')

# Save the statistical plot with proper spacing
plt.tight_layout()
plt.savefig('exponentiation_statistics.png', dpi=300, bbox_inches='tight')
plt.close()

# Clean up GPU memory
cp.get_default_memory_pool().free_all_blocks()