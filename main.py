import functools
import numpy as np
import os
import pandas as pd
from pathlib import Path
import psutil
import random
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import time
import tracemalloc
from typing import Optional, Callable, Any
from xgboost import XGBRegressor
import functools
import time
import psutil
import tracemalloc
import threading
import gc
import inspect
from typing import Callable, Any
import numpy as np
from multiprocessing import Pool
import numpy as np
import numpy as np
from multiprocessing import Pool
import multiprocessing
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from joblib import Parallel, delayed
import numpy as np
from joblib import Parallel, delayed
import time
from tabulate import tabulate
import pandas as pd
import cupy as cp

################################################################################
#
# performance evaluation function
#
################################################################################

def performance_monitor(func: Callable) -> Callable:
    """
    A decorator that monitors and reports comprehensive function performance metrics including:
    - Execution time
    - CPU usage
    - Memory usage and peak memory
    - Thread utilization
    - I/O operations
    - Context switches
    - Page faults
    - Network usage
    - Garbage collection statistics
    - Stack depth

    Args:
        func: The function to be monitored

    Returns:
        Wrapper function that includes performance monitoring
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Initialize process monitoring
        process = psutil.Process()

        # Start collecting initial metrics
        initial_cpu_time = process.cpu_percent()
        initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB

        try:
            initial_io = process.io_counters()
            initial_ctx = process.num_ctx_switches()
            initial_faults = process.memory_full_info().num_page_faults
            initial_net = psutil.net_io_counters()
            initial_gc_count = gc.get_count()
        except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
            print(f"Warning: Some metrics may be unavailable - {str(e)}")
            initial_io = initial_ctx = initial_faults = initial_net = None

        # Start memory tracking
        tracemalloc.start()
        start_time = time.perf_counter()

        # Execute the function
        result = func(*args, **kwargs)

        # Collect final metrics
        end_time = time.perf_counter()
        final_cpu_time = process.cpu_percent()
        final_memory = process.memory_info().rss / 1024 / 1024

        # Get memory tracking results
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Thread metrics
        thread_count = process.num_threads()
        thread_utilization = len(threading.enumerate()) / thread_count

        # Calculate final metrics
        try:
            if initial_io:
                final_io = process.io_counters()
                read_bytes = final_io.read_bytes - initial_io.read_bytes
                write_bytes = final_io.write_bytes - initial_io.write_bytes

            if initial_ctx:
                final_ctx = process.num_ctx_switches()
                voluntary_ctx_switches = final_ctx.voluntary - initial_ctx.voluntary
                involuntary_ctx_switches = final_ctx.involuntary - initial_ctx.involuntary

            if initial_faults:
                final_faults = process.memory_full_info().num_page_faults
                page_faults = final_faults - initial_faults

            if initial_net:
                final_net = psutil.net_io_counters()
                bytes_sent = final_net.bytes_sent - initial_net.bytes_sent
                bytes_received = final_net.bytes_recv - initial_net.bytes_recv

            if initial_gc_count:
                final_gc_count = gc.get_count()
                gc_runs = tuple(
                    f - i for f, i in zip(final_gc_count, initial_gc_count))

        except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
            print(f"Warning: Some metrics may be unavailable - {str(e)}")

        # Get stack depth
        stack_depth = len(inspect.stack())

        # Calculate basic metrics
        execution_time = end_time - start_time
        memory_used = final_memory - initial_memory
        cpu_usage = final_cpu_time - initial_cpu_time

        # Print performance report
        print(f"\nPerformance Metrics for {func.__name__}:")
        print(f"{'=' * 70}")

        # Basic metrics
        print(f"Execution Time: {execution_time:.4f} seconds")
        print(f"CPU Usage: {cpu_usage:.2f}%")
        print(f"Memory Usage: {memory_used:.2f} MB")
        print(f"Peak Memory: {peak / 1024 / 1024:.2f} MB")

        # Thread metrics
        print(f"Thread Count: {thread_count}")
        print(f"Thread Utilization: {thread_utilization:.2%}")

        # I/O metrics
        if initial_io:
            print(
                f"I/O Operations - Read: {read_bytes / 1024:.2f} KB, Write: {write_bytes / 1024:.2f} KB")

        # Context switch metrics
        if initial_ctx:
            print(
                f"Context Switches - Voluntary: {voluntary_ctx_switches}, Involuntary: {involuntary_ctx_switches}")

        # Page fault metrics
        if initial_faults:
            print(f"Page Faults: {page_faults}")

        # Network metrics
        if initial_net:
            print(
                f"Network Usage - Sent: {bytes_sent / 1024:.2f} KB, Received: {bytes_received / 1024:.2f} KB")

        # GC metrics
        if initial_gc_count:
            print(f"GC Runs (Gen 0/1/2): {gc_runs}")

        # Stack metrics
        print(f"Stack Depth: {stack_depth}")
        print(f"{'=' * 70}\n")

        return result

    return wrapper

################################################################################
#
# base model training function
#
# Features INCLUDED:
#
# 'trip_distance' - The actual distance of the trip in miles
# 'hour' - Hour of the day (0-23) extracted from pickup time
# 'day_of_week' - Day of week (0-6, where 0 is Monday) from pickup time
# 'month' - Month (1-12) from pickup time
# 'passenger_count' - Number of passengers in the taxi
# 'PULocationID' - Pickup location zone ID
# 'DOLocationID' - Dropoff location zone ID
#
# Features NOT included but available in typical taxi data:
#
# 'tpep_pickup_datetime' - Raw pickup timestamp (instead we extracted hour/day/month)
# 'tpep_dropoff_datetime' - Raw dropoff timestamp
# 'total_amount' - Total paid (includes fare, tips, tolls)
# 'tip_amount' - Tip paid
# 'tolls_amount' - Toll charges
# 'mta_tax' - MTA tax charged
# 'extra' - Extra charges
# 'improvement_surcharge' - Improvement surcharge fee
# 'RatecodeID' - Rate type of the trip
# 'store_and_fwd_flag' - Store and forward flag
# 'payment_type' - How the passenger paid
#
# Target Variable:
#
# 'fare_amount' - The base fare cost (what we're trying to predict)
#
# The reasoning behind these choices:
#
# We chose features that would be known BEFORE the trip starts (to make useful predictions)
# We broke down datetime into components (hour/day/month) because they're more useful for the model than raw timestamps
# We excluded post-trip information like tips and final amounts
# We excluded administrative fields like store_and_fwd_flag
# Location IDs are included because they can capture factors like typical traffic patterns and route characteristics for different areas
#
################################################################################

@performance_monitor
def train_taxi_fare_model(taxi_df: pd.DataFrame) -> tuple:
    """
    Train XGBoost model on taxi data and return model and metrics.

    Args:
        taxi_df: DataFrame containing taxi ride data

    Returns:
        tuple: (trained_model, metrics_dict, training_time)
    """
    start_time = time.time()

    # Feature engineering
    taxi_df = taxi_df.copy()
    taxi_df['hour'] = taxi_df['tpep_pickup_datetime'].dt.hour
    taxi_df['day_of_week'] = taxi_df['tpep_pickup_datetime'].dt.dayofweek
    taxi_df['month'] = taxi_df['tpep_pickup_datetime'].dt.month

    # Select features
    features = [
        'trip_distance',
        'hour',
        'day_of_week',
        'month',
        'passenger_count',
        'PULocationID',
        'DOLocationID'
    ]

    # Prepare data
    X = taxi_df[features]
    y = taxi_df['fare_amount']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'training_samples': len(X_train),
        'testing_samples': len(X_test)
    }

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Calculate training time
    training_time = time.time() - start_time

    # Print summary
    print("\nModel Training Summary:")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Number of training samples: {metrics['training_samples']}")
    print(f"Number of testing samples: {metrics['testing_samples']}")
    print(f"\nPerformance Metrics:")
    print(f"RMSE: ${metrics['rmse']:.2f}")
    print(f"MAE: ${metrics['mae']:.2f}")
    print(f"R² Score: {metrics['r2']:.3f}")
    print("\nTop 5 Most Important Features:")
    print(importance_df.head().to_string(index=False))

    return model, metrics, training_time

################################################################################
#
# perform basic arithmetic testing using different data structures
#
################################################################################

size = int(1e8)

# basic exponentiation function
@performance_monitor
def exponentiate(base, power):
    base ** power
    return

# performance with list type object
list_vec = np.random.randint(1, 101, size=size, dtype=np.int64)

# see size of list_vev in memory
bytes_size = sys.getsizeof(list_vec)
megabytes = bytes_size / (1024 * 1024)
print(f"{megabytes:.2f} MB")

# for loop must be used for list type object
@performance_monitor
def for_loop_exponentiate(list_vec: list, power: int):
    [x ** power for x in list_vec]
    return

#for_loop_exponentiate(list_vec, 100)

# performance with numpy type array, with single core (defualt)
numpy_arr = np.array(list_vec)

exponentiate(numpy_arr, 100)

# performance with pandas type series
pandas_ser = pd.Series(list_vec)

exponentiate(pandas_ser, 100)

# using np.power isntead of standard python operator
def exponentiate_np_power(base, power):
    # Use numpy's power function - it's highly optimized for arrays
    return np.power(base, power)

performance_monitor(exponentiate_np_power)(numpy_arr, 100)

# multithreaded with numpy
max_threads = multiprocessing.cpu_count()
print(f"Logical cores: {max_threads}")


def exp_chunk(chunk, power):
    return np.power(chunk, power)

@performance_monitor
def parallel_exponentiate(array, power, n_jobs=max_threads):
    chunks = np.array_split(array, n_jobs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(exp_chunk)(chunk, power) for chunk in chunks
    )

    #return np.concatenate(results)
    return

parallel_exponentiate(numpy_arr, 100)


@performance_monitor
def cupy_exponential(arr, power):
    # Transfer to GPU
    gpu_arr = cp.asarray(arr)
    # Compute on GPU
    result = cp.power(gpu_arr, power)
    # Transfer back to CPU
    cp.asnumpy(result)
    return


# Example usage
cupy_exponential(numpy_arr, 100)



################################################################################
#
# performance using raw pandas for 1 year of data
#
################################################################################

# read in parquet file into pd data frame
# df_day = pd.read_parquet('./data/yellow_tripdata_2023_01_01.parquet')
# df_month = pd.read_parquet('./data/yellow_tripdata_2023_01.parquet')
# df_year = pd.read_parquet('./data/yellow_tripdata_2023.parquet')
#
#
#
#
#
#
#
#
#
# df_day["fare_amount"].dtype
#
#
# @performance_monitor
# def multiply_fare(multiple: int) -> int:
#     return df_day["fare_amount"] * multiple
#
# result = multiply_fare(1024102410241024)
# print(f"Result of addition: {result}")
#
# model = train_taxi_fare_model(df_year)
#
# # Create equivalent data structures
# size = 1000000
# list_vec = list(range(size))
# numpy_arr = np.array(list_vec)
# pandas_ser = pd.Series(list_vec)
#
# # Test multiplication operation
# def test_speed():
#     # List comprehension
#     start = time.time()
#     [x * 2 for x in list_vec]
#     list_time = time.time() - start
#
#     # NumPy array
#     start = time.time()
#     numpy_arr * 2
#     numpy_time = time.time() - start
#
#     # Pandas Series
#     start = time.time()
#     pandas_ser * 2
#     pandas_time = time.time() - start
#
#     return list_time, numpy_time, pandas_time
#
# test_speed()

################################################################################
#
# end of main.py
#
################################################################################
