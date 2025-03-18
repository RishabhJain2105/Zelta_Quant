import os
import pandas as pd
import numpy as np
import uuid
import matplotlib.pyplot as plt
from untrade.client import Client
import math
import requests


def process_data(data):
    calculate_rsi = lambda data, window=14: 100 - (
        100 / (
            1 + (
                (data['closer'].diff(1).where(data['closer'].diff(1) > 0, 0).rolling(window=window).mean()) /
                (-data['closer'].diff(1).where(data['closer'].diff(1) < 0, 0).rolling(window=window).mean())
            )
        )
    )

    calculate_bollinger_bands = lambda data, window=20, num_std_dev=2: (
        data['closer'].rolling(window=window).mean() +
        (data['closer'].rolling(window=window).std() * num_std_dev),
        data['closer'].rolling(window=window).mean(),
        data['closer'].rolling(window=window).mean() -
        (data['closer'].rolling(window=window).std() * num_std_dev)
    )
    calculate_sma = lambda data, window: data['close'].rolling(window=window, min_periods=1).mean()

    data['datetime'] = pd.to_datetime(data['datetime'])
    time_diff = (data['datetime'].iloc[1] - data['datetime'].iloc[0]).total_seconds()
    a, b = 100.3, 0.0001259
    sma_window = max(1, math.ceil(a * np.exp(-b * time_diff)))  # Ensure sma_window is at least 1

    data['closer'] = calculate_sma(data, window=sma_window)
    
    # Convert time difference to days
    time_diff_days = time_diff / 1440 *60  # 1440 minutes in 1 day
    # Define desired periods in days
    desired_period_30_days = 30  # 30 days
    desired_period_150_days = 150  # 150 days

    # Convert periods to minutes
    period_30_days_in_minutes = desired_period_30_days * 24 * 60*60
    period_150_days_in_minutes = desired_period_150_days * 24 * 60*60
    # Convert periods to minutes
    # Define desired periods in days for RSI and Bollinger Bands
    desired_period_rsi = 14  # 14 days
    desired_period_bbands = 20  # 20 days

    reference_time_diff = 4 * 60 * 60  # 4 hours in seconds
    period_bbands_in_minutes = desired_period_bbands * 24 * 60*60

    # Calculate dynamic windows
    window_rsi = max(1, int(14*reference_time_diff/ time_diff))
    window_bbands = max(1, int(20*reference_time_diff / time_diff))

    if window_rsi < 1:
        print("Adjusting RSI window")
        window_rsi = max(1, int(desired_period_rsi / time_diff_days))
    if window_bbands < 1:
        print("Adjusting Bollinger Bands window")
        window_bbands = max(1, int(desired_period_bbands / time_diff_days))

    # Calculate dynamic spans
    # Calculate dynamic spans
    span_30 = period_30_days_in_minutes / time_diff  # Ensure span is at least 1
    span_150 = period_150_days_in_minutes / time_diff  # Ensure span is at least 1
    if span_30 < 1 :
        print("Less than 1:30")
        span_30 = max(desired_period_30_days / time_diff_days, 1)
    if span_150 < 1 :
        print("Less than 1:150")
        span_150 = max(desired_period_150_days / time_diff_days, 1)
    # Apply EMA using the dynamically calculated spans
    data['30_period_EMA'] = data['closer'].ewm(span=span_30, adjust=False).mean()
    data['150_period_EMA'] = data['closer'].ewm(span=span_150, adjust=False).mean()
     # Apply RSI and Bollinger Bands with dynamic windows
    data['RSI'] = calculate_rsi(data, window=window_rsi)
    data['upper_band'], data['middle_band'], data['lower_band'] = calculate_bollinger_bands(data, window=window_bbands)

    # data['ATR'] = calculate_atr(data)
    return data

# Strategy logic implementation
def strat(data):
    signals = []
    position = 0
    # Before atr
    stop_loss_pct, take_profit_pct = 0.001, 0.003
    entry_price, sposition = 0, None
    data['trade_type'] = None
    # stop_loss_multiplier = 1  # Multiplier for ATR
    # take_profit_multiplier = 20  # Multiplier for ATR

    for i in range(1, len(data)):
        if sposition is not None and position != 0:
            if sposition == -2 and ((data['closer'][i] > entry_price * (1 + stop_loss_pct)) or (data['closer'][i] < entry_price * (1 - take_profit_pct))):
                sposition = None
                signals.append(2)
                data.at[i, 'trade_type'] = 'long_reversal'
                entry_price = 0
                continue
            elif sposition == 2 and ((data['closer'][i] < entry_price * (1 - stop_loss_pct)) or (data['closer'][i] > entry_price * (1 + take_profit_pct))):
                sposition = None
                signals.append(-2)
                data.at[i, 'trade_type'] = 'short_reversal'
                entry_price = 0
                continue
            else:
                signals.append(0)
                continue
        
        if sposition is not None and position == 0:
            if sposition == -2 and ((data['closer'][i] > entry_price * (1 + stop_loss_pct)) or (data['closer'][i] < entry_price * (1 - take_profit_pct))):
                sposition = None
                signals.append(1)
                data.at[i, 'trade_type'] = 'close'
                entry_price = 0
                continue
            elif sposition == 2 and ((data['closer'][i] < entry_price * (1 - stop_loss_pct)) or (data['closer'][i] > entry_price * (1 + take_profit_pct))):
                sposition = None
                signals.append(-1)
                data.at[i, 'trade_type'] = 'close'
                entry_price = 0
                continue
            elif data['30_period_EMA'][i] > data['150_period_EMA'][i] and data['closer'][i] > data['30_period_EMA'][i]:
                signals.append(2)
                position = 1
                sposition = None
                entry_price = data['close'][i]
                data.at[i, 'trade_type'] = 'long_reversal'
            elif data['30_period_EMA'][i] < data['150_period_EMA'][i] and data['closer'][i] < data['30_period_EMA'][i]:
                signals.append(-2)
                position = -1
                sposition = None
                entry_price = data['close'][i]
                data.at[i, 'trade_type'] = 'short_reversal'
            else:
                signals.append(0)
                
        if position == 0 and sposition is None:
            if data['30_period_EMA'][i] > data['150_period_EMA'][i] and data['closer'][i] > data['30_period_EMA'][i]:
                signals.append(1)
                position = 1
                entry_price = data['close'][i]
                data.at[i, 'trade_type'] = 'long'
            elif data['30_period_EMA'][i] < data['150_period_EMA'][i] and data['closer'][i] < data['30_period_EMA'][i]:
                signals.append(-1)
                position = -1
                entry_price = data['close'][i]
                data.at[i, 'trade_type'] = 'short'
            elif data['RSI'][i] < 90 and data['closer'][i] < data['lower_band'][i]:
                signals.append(-1)
                data.at[i, 'trade_type'] = 'short'
                sposition = -2
                entry_price = data['close'][i]
            elif data['RSI'][i] > 10 and data['closer'][i] > data['upper_band'][i]:
                signals.append(1)
                data.at[i, 'trade_type'] = 'long'
                sposition = 2
                entry_price = data['close'][i]
            else:
                signals.append(0)

        elif position == 1:
            if data['RSI'][i] > 70 and data['closer'][i] > data['upper_band'][i]:
                signals.append(-2)
                data.at[i, 'trade_type'] = 'short_reversal'
                sposition = -2
                entry_price = data['close'][i]
            elif data['closer'][i] < data['30_period_EMA'][i]:
                signals.append(-1)
                data.at[i, 'trade_type'] = 'close'
                position = 0
                entry_price = 0
            else:
                signals.append(0)

        elif position == -1:
            if data['RSI'][i] < 30 and data['closer'][i] < data['lower_band'][i]:
                signals.append(2)
                data.at[i, 'trade_type'] = 'long_reversal'
                sposition = 2
                entry_price = data['close'][i]
            elif data['closer'][i] > data['30_period_EMA'][i]:
                signals.append(1)
                data.at[i, 'trade_type'] = 'close'
                position = 0
                entry_price = 0
            else:
                signals.append(0)

    # Ensure the signals list matches the DataFrame length
    signals = [0] + signals  # Prepend a zero for the first row
    signals = signals[:len(data)]  # Truncate signals if too long
    while len(signals) < len(data):  # Pad with zeros if too short
        signals.append(0)

    data['signals'] = signals
    return data

# Following function can be used for every size of file, specially for large files(time consuming,depends on upload speed and file size)
def perform_backtest(csv_file_path):
    """
    Perform a backtest for small-sized files.

    Parameters:
    csv_file_path (str): Path to the CSV file containing backtest data.

    Returns:
    dict or list: Backtest results.
    """
    client = Client()
    try:
        response = client.backtest(
            jupyter_id="test",  # Replace with your actual user ID
            file_path=csv_file_path,
            leverage=1,  # Adjust leverage as needed
            result_type="Q",
        )
        
        # Print the raw response content for debugging
        print("Raw Response:", response.text)  # This will show the actual content returned by the server
        
        # If the response is empty, handle gracefully
        if not response.text.strip():
            print("Error: The response is empty or invalid JSON.")
            return None
        
        # Attempt to parse the JSON response
        result = response.json()  # Try parsing the JSON response
        return result
    
    except requests.exceptions.JSONDecodeError as e:
        print("Error decoding JSON response:", e)
        print("Response body:", response.text)  # Print response body to check what went wrong
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None

def perform_backtest_large_csv(csv_file_path):
    """
    Perform a backtest for large files using chunked uploads.

    Parameters:
    csv_file_path (str): Path to the CSV file.

    Returns:
    list or dict: Backtest results.
    """
    client = Client()
    file_id = str(uuid.uuid4())
    chunk_size = 90 * 1024 * 1024  # 90 MB chunks
    total_size = os.path.getsize(csv_file_path)
    total_chunks = (total_size + chunk_size - 1) // chunk_size
    chunk_number = 0
    result = []

    if total_size <= chunk_size:
        # Normal Backtest for small files
        result = client.backtest(
            file_path=csv_file_path,
            leverage=1,
            jupyter_id="test",
            result_type="Q",
        )
        for value in result:
            print(value)
        return result

    with open(csv_file_path, "rb") as f:
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            chunk_file_path = f"/tmp/{file_id}chunk{chunk_number}.csv"
            with open(chunk_file_path, "wb") as chunk_file:
                chunk_file.write(chunk_data)

            # Large CSV Backtest
            result_chunk = client.backtest(
                file_path=chunk_file_path,
                leverage=1,
                jupyter_id="test",
                file_id=file_id,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                result_type="Q",
            )

            for value in result_chunk:
                print(value)
                result.append(value)

            os.remove(chunk_file_path)
            chunk_number += 1

    return result

def perform_backtest(csv_file_path):
    client = Client()
    try:
        response = client.backtest(
            jupyter_id="test",  # Replace with your actual user ID
            file_path=csv_file_path,
            leverage=1,  # Adjust leverage as needed
            result_type="Q",
        )
        
        # Print the raw response content for debugging
        print("Raw Response:", response.text)  # This will show the actual content returned by the server
        
        # If the response is empty, handle gracefully
        if not response.text.strip():
            print("Error: The response is empty or invalid JSON.")
            return None
        
        # Attempt to parse the JSON response
        result = response.json()  # Try parsing the JSON response
        return result
    
    except requests.exceptions.JSONDecodeError as e:
        print("Error decoding JSON response:", e)
        print("Response body:", response.text)  # Print response body to check what went wrong
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None


def main():
    """
    Process a single CSV file, apply the strategy, and perform backtesting.
    """
    # Load data
    data = pd.read_csv("d/BTC_2019_2023_15m.csv")

    
    # Process the data
    processed_data = process_data(data)
    
    # Generate signals
    result_data = strat(processed_data)
    
    # Save results to CSV
    csv_file_path = "results.csv" 
    result_data.to_csv(csv_file_path, index=False)
    
    # Perform backtest
    backtest_result = perform_backtest_large_csv(csv_file_path) 
    print(backtest_result)
    for value in backtest_result:
        print(value)
    print("Backtest results for ",csv_file_path)
    

if __name__ == "__main__":
    main()