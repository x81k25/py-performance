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
# end of exponentiation.py
#
################################################################################
