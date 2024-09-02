import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import os
import multiprocessing as mp
import pandas_ta as ta
from itertools import product
from tqdm import tqdm
import csv
import datetime as dt

pairs = [
    "SPX500_USD",
    "UK100_GBP",
    "DE30_EUR",
    "XAU_USD",
    "BCO_USD",
    "CORN_USD",
    "XCU_USD",
    "US2000_USD",
    "XAG_USD",
    "GBP_JPY",
    "EUR_USD",
    "GBP_USD",
    "EUR_USD",
]
granualrity = ["M15", "M30", "H1"]
years = [2023, 2024]
FILE_NAME = "KeltnerChannelBreakout"
FOLDER_NAME = "discovery_testing"

# Placeholder function to calculate the 200-period SMA


def isNowInTimePeriod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else:
        # Over midnight:
        return nowTime >= startTime or nowTime <= endTime


def construct_df_O(commodity, granualrity, min_year=2023, max_year=None, DIRECTORY_PATH='./data'):
    if os.path.isdir(DIRECTORY_PATH):
        path = DIRECTORY_PATH
    elif os.path.isdir('./data'):
        path = "./data"
    elif os.path.isdir("../data"):
        path = "../data"
    else:
        path = "../../data"

    df = pd.read_pickle(f"{path}/{commodity}_{granualrity}.pkl")

    df.rename(
        columns={
            "time": "Time",
            "volume": "Volume",
            "mid_o": "Open",
            "mid_h": "High",
            "mid_c": "Close",
            "mid_l": "Low",
            "bid_o": "B_Open",
            "bid_h": "B_High",
            "bid_c": "B_Close",
            "bid_l": "B_Low",
            "ask_o": "A_Open",
            "ask_h": "A_High",
            "ask_c": "A_Close",
            "ask_l": "A_Low",
        },
        inplace=True,
    )

    df["Time"] = pd.to_datetime(df["Time"])

    df = df[
        (df["Time"] > str(min_year))
        & (df["Time"] < str(min_year + 1 if not max_year else max_year))
    ]

    df = df[["Time", "Volume", "High", "Close", "Open", "Low"]]
    df.reset_index(drop=True, inplace=True)

    return df


def calculate_sma(data, period=200):
    return ta.sma(pd.Series(data), period)


# Placeholder function to calculate the 200-period EMA
def calculate_ema(data, period=200):
    return ta.ema(pd.Series(data), period)


# Custom implementation of the MACD
def MACD(series, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = calculate_ema(series, fast_period)
    slow_ema = calculate_ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# Custom implementation of the Average Directional Index (ADX)
def ADX(high, low, close, period=14):
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    tr = np.maximum(
        high - low, np.maximum(abs(high - close.shift()),
                               abs(low - close.shift()))
    )
    atr = tr.rolling(window=period).mean()

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()

    return adx


# Placeholder function to calculate the Keltner Channels
def calculate_keltner_channels(high, low, close, length=40):
    typical_price = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
    ema = typical_price.ewm(span=length, adjust=False).mean()
    atr = pd.Series(high).sub(pd.Series(low)).ewm(
        span=length, adjust=False).mean()
    upper_band = ema + (atr * 2)
    lower_band = ema - (atr * 2)
    return upper_band, ema, lower_band


# Placeholder function to calculate the 25-period Hull Moving Average
def calculate_hma(data, period=25):
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))
    wma_half = 2 * pd.Series(data).rolling(window=half_length).mean()
    wma_full = pd.Series(data).rolling(window=period).mean()
    raw_hma = wma_half - wma_full
    hma = raw_hma.rolling(window=sqrt_length).mean()
    return hma


# Placeholder function to calculate the calculate_rsi and its 20-period calculate_sma
def calculate_rsi(data, rsi_length=14, rsi_sma_length=20):
    delta = pd.Series(data).diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_length).mean()
    avg_loss = loss.rolling(window=rsi_length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_sma = rsi.rolling(window=rsi_sma_length).mean()
    return rsi, rsi_sma

# Custom implementation of the MACD


def MACD(series, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = calculate_ema(series, fast_period)
    slow_ema = calculate_ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def get_atr(high, low, close, length):
    return ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length)

# Define the strategy class


class KeltnerChannelBreakoutStrategy(Strategy):
    # Define parameters for the strategy (can be optimized later)
    keltner_length = 40
    rsi_length = 14
    rsi_sma_length = 20
    hma_length = 25
    ma_length = 200
    units = 0.1  # Position size
    macd_enabled = False
    adx_enabled = False
    atr_filter_enabled = False
    partial_profit_enabled = False
    atr_threshold = 0.5  # ATR threshold for volatility filter
    partial_profit_level = 0.5  # Level at which to take partial profits
    atr_length = 14

    def init(self):
        # Calculate indicators
        self.ma_200 = self.I(calculate_sma, self.data.Close, self.ma_length)
        self.rsi, self.rsi_sma = self.I(
            calculate_rsi, self.data.Close, self.rsi_length, self.rsi_sma_length)
        self.hma = self.I(calculate_hma, self.data.Close, self.hma_length)
        self.upper_keltner, self.middle_keltner, self.lower_keltner = self.I(
            calculate_keltner_channels, self.data.High, self.data.Low, self.data.Close, self.keltner_length)

        if self.macd_enabled:
            self.macd_line, self.signal_line, _ = self.I(MACD, self.data.Close)

        if self.adx_enabled:
            self.adx = self.I(ADX, self.data.High,
                              self.data.Low, self.data.Close)

        if self.atr_filter_enabled:
            self.atr = self.I(get_atr, self.data.High,
                              self.data.Low, self.data.Close, self.atr_length)

    def next(self):
        date_time = pd.to_datetime(self.data.Time[-1])
        day = date_time.day
        month = date_time.month
        year = date_time.year
        hour = date_time.hour
        minute = date_time.minute

        can_trade = isNowInTimePeriod(
            dt.time(9, 0), dt.time(21, 0), dt.time(hour, minute)
        )

        if not can_trade:
            return

        # Check for MACD and ADX conditions if enabled
        macd_condition = (not self.macd_enabled) or (
            self.macd_line[-1] > self.signal_line[-1])
        adx_condition = (not self.adx_enabled) or (self.adx[-1] > 25)

        # Volatility Filter
        if self.atr_filter_enabled and self.atr[-1] < self.atr_threshold:
            return

        # Long Entry Conditions
        if (self.data.Close[-1] > self.ma_200[-1] and
            self.data.Close[-1] > self.hma[-1] and
            self.rsi[-1] > self.rsi_sma[-1] and
            self.data.Close[-1] > self.upper_keltner[-1] and
            macd_condition and
                adx_condition):
            stop_loss = self.data.Low[-2:].min()
            self.buy(sl=stop_loss, size=self.units)

        # Short Entry Conditions
        elif (self.data.Close[-1] < self.ma_200[-1] and
              self.data.Close[-1] < self.hma[-1] and
              self.rsi[-1] < self.rsi_sma[-1] and
              self.data.Close[-1] < self.lower_keltner[-1] and
              macd_condition and
              adx_condition):
            stop_loss = self.data.High[-2:].max()

            self.sell(sl=stop_loss, size=self.units)

        # Manage open trades
        for index in range(len(self.trades)):
            position = self.trades[index]
            # Dynamic Stop Loss
            if position.is_long:
                stop_loss = max(self.data.Low[-2:])
                if self.data.Close[-1] < stop_loss:
                    position.close()
                elif self.partial_profit_enabled:
                    if self.data.Close[-1] >= self.upper_keltner[-1] * (1 + self.partial_profit_level):
                        position.close(size=self.units / 2)

            if position.is_short:
                stop_loss = min(self.data.High[-2:])
                if self.data.Close[-1] > stop_loss:
                    position.close()
                elif self.partial_profit_enabled:
                    if self.data.Close[-1] <= self.lower_keltner[-1] * (1 - self.partial_profit_level):
                        position.close(size=self.units / 2)

# Function to generate all combinations of options and parameter ranges


def run_backtest(params):
    pair, granualrity = params

    param_ranges = {
        "macd_enabled": [True, False],
        "adx_enabled": [True, False],
        "atr_filter_enabled": [True, False],
        "partial_profit_enabled": [True, False],
        "keltner_length": [20, 40, 60],
        "rsi_length": [14, 20],
        "rsi_sma_length": [10, 20],
        "hma_length": [25, 50],
        "ma_length": [100, 200],
        "atr_length": [14, 20],
        "atr_threshold": [0.5, 1.0],
        "partial_profit_level": [0.5]
    }

    # Generate all parameter combinations
    param_combinations = list(product(*param_ranges.values()))
    df = construct_df_O(pair, granualrity, 2022, 2024)

    for param_combination in param_combinations:
        param_dict = dict(zip(param_ranges.keys(), param_combination))

        # Run the backtest
        bt = Backtest(
            df,
            KeltnerChannelBreakoutStrategy,
            cash=5000,
            commission=0.002,
            margin=1 / 50,
        )
        stats = bt.run(**param_dict)

        # Prepare the result dictionary
        result = [
            pair,
            granualrity,
            param_dict["keltner_length"],     # [20, 40, 60]
            param_dict["ma_length"],          # [100, 200]
            param_dict["rsi_length"],         # [14, 20]
            param_dict["hma_length"],         # [25, 50]
            param_dict["macd_enabled"],       # [True, False]
            param_dict["adx_enabled"],        # [True, False]
            param_dict["atr_filter_enabled"],  # [True, False]
            param_dict["atr_length"],         # [14, 20]
            param_dict["partial_profit_enabled"],  # [True, False]
            stats["Return [%]"],
            stats["Equity Final [$]"],
            stats["Sharpe Ratio"],
            stats["Max. Drawdown [%]"],
            stats["Win Rate [%]"],
            stats["# Trades"],
        ]

        save_result_to_csv(result)

# Function to save a single result to CSV


def save_result_to_csv(result):
    csv_file = FILE_NAME + "_" + f"optimization_results.csv"

    write_header = not pd.io.common.file_exists(csv_file)
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(
                [
                    "Pair",
                    "Granularity",
                    "Keltner Length",
                    "MA Length",
                    "RSI Length",
                    "HMA Length",
                    "MACD Enabled",
                    "ADX Enabled",
                    "ATR Filter Enabled",
                    "ATR Length",
                    "Partial Profit Enabled",
                    "Return [%]",
                    "Equity Final [$]",
                    "Sharpe Ratio",
                    "Max. Drawdown [%]",
                    "Win Rate [%]",
                    "Total Trades",
                ]
            )
        writer.writerow(result)

# Main function to set up multiprocessing


def main():
    print(f"** Starting {FILE_NAME} script **")
    trade_combinations = list(product(pairs, granualrity))

    pool = mp.Pool(mp.cpu_count())  # Use all available cores
    for _ in tqdm(pool.imap_unordered(run_backtest, trade_combinations), total=len(trade_combinations)):
        pass  # The progress bar will update with each completed task

    pool.close()
    pool.join()

    print(f"Optimization completed. Results saved.")


# Use multiprocessing to run the backtests in parallel
if __name__ == "__main__":
    main()
