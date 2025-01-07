import pandas as pd
from pathlib import Path
import requests
from typing import Optional

################################################################################
#
# download and store taxi data
#
################################################################################

def download_taxi_data(
    year: int,
    month: Optional[int] = None,
    day: Optional[int] = None,
    data_type: str = 'yellow',
    retry_attempts: int = 3
) -> Optional[pd.DataFrame]:
    """
    Downloads taxi data for a specified timeframe (year, month, or day).

    Args:
        year: The year of data
        month: Optional; The month of data (1-12)
        day: Optional; The specific day to extract (1-31)
        data_type: The type of taxi data ('yellow' or 'green')
        retry_attempts: Number of download retry attempts

    Returns:
        DataFrame containing requested taxi data, or None if download fails
    """
    if month is not None:
        month = str(month).zfill(2)
        filename = f"{data_type}_tripdata_{year}-{month}.parquet"
    else:
        # If no month specified, create a list of all months
        months = [str(m).zfill(2) for m in range(1, 13)]
        dfs = []

        for m in months:
            df = download_taxi_data(year, int(m), day, data_type,
                                    retry_attempts)
            if df is not None:
                dfs.append(df)
            else:
                print(f"Failed to download data for {year}-{m}")

        return pd.concat(dfs) if dfs else None

    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    url = base_url + filename

    for attempt in range(retry_attempts):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Create temporary file path
            temp_path = Path(f"temp_{filename}")

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0

            print(f"Downloading {filename}...")
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percent = downloaded / total_size * 100
                        print(f"\rProgress: {percent:.1f}%", end='')

            print("\nLoading data...")

            # Read the parquet file
            df = pd.read_parquet(temp_path)
            df['tpep_pickup_datetime'] = pd.to_datetime(
                df['tpep_pickup_datetime'])

            # Filter based on day if specified
            if day is not None:
                df = df[df['tpep_pickup_datetime'].dt.day == day].copy()
                print(f"Extracted data for {year}-{month}-{day}")
            else:
                print(f"Extracted data for {year}-{month}")

            print(f"Number of records: {len(df)}")

            # Remove temporary file
            temp_path.unlink()

            return df

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < retry_attempts - 1:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(
                    f"Failed to download data after {retry_attempts} attempts")
                return None
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
            return None

# get and save year of data
df_year = download_taxi_data(2023)
df_year.to_parquet(Path("data/yellow_tripdata_2023.parquet"))

# get and save month of data
df_month = download_taxi_data(2023, 6)
df_month.to_parquet(Path("data/yellow_tripdata_2023_01.parquet"))

# get and save day of data
df_day = download_taxi_data(2023, 6, 15)
df_day.to_parquet(Path("data/yellow_tripdata_2023_01_01.parquet"))

################################################################################
#
# end of download_data.py
#
################################################################################
