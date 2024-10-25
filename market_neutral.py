import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import yfinance as yf


# Fetch historical data for two cryptocurrencies
def fetch_data(crypto1, crypto2, start, end):
    data1 = yf.download(crypto1, start=start, end=end)['Close']
    data2 = yf.download(crypto2, start=start, end=end)['Close']
    return data1, data2

# Calculate the spread
def calculate_spread(data1, data2):
    model = sm.OLS(data1, sm.add_constant(data2)).fit()
    spread = data1 - model.predict(sm.add_constant(data2))
    return spread

# Generate signals with entry and exit criteria
def generate_signals(spread, window=30, entry_z=1.0, exit_z=0.0):
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    z_score = (spread - mean) / std

    signals = np.zeros(len(spread))
    signals[z_score > entry_z] = -1  # Short signal
    signals[z_score < -entry_z] = 1  # Long signal
    signals[z_score < exit_z] = 0   # Exit signal

    return signals

# Backtesting the strategy with transaction costs and risk management
def backtest(data1, data2, signals, transaction_cost=0.001, stop_loss=0.02, take_profit=0.03):
    position = 0
    cumulative_returns = []

    for i in range(1, len(signals)):
        # If we have a position
        if position != 0:
            # Check for exit conditions
            if (position == 1 and (data1[i] / data1[i-1] - 1) >= take_profit) or \
               (position == -1 and (data2[i] / data2[i-1] - 1) >= take_profit):
                position = 0  # Exit position
            elif (position == 1 and (data1[i] / data1[i-1] - 1) <= -stop_loss) or \
                 (position == -1 and (data2[i] / data2[i-1] - 1) <= -stop_loss):
                position = 0  # Stop loss hit

        # Update position based on signals
        if signals[i] == 1 and position == 0:  # Long signal
            position = 1
            cumulative_returns.append(-transaction_cost)  # Cost of entering position
        elif signals[i] == -1 and position == 0:  # Short signal
            position = -1
            cumulative_returns.append(-transaction_cost)  # Cost of entering position
        else:
            cumulative_returns.append(0)  # No cost if no trade

        # Calculate daily returns
        if position == 1:
            daily_return = data1[i] / data1[i-1] - 1
            cumulative_returns[-1] += daily_return
        elif position == -1:
            daily_return = data2[i] / data2[i-1] - 1
            cumulative_returns[-1] += daily_return

    return np.cumsum(cumulative_returns)

# Main function
def main():
    crypto1, crypto2 = 'BTC-USD', 'ETH-USD'
    start_date, end_date = '2024-04-01', '2024-10-01'

    data1, data2 = fetch_data(crypto1, crypto2, start_date, end_date)
    spread = calculate_spread(data1, data2)
    signals = generate_signals(spread)

    cumulative_returns = backtest(data1, data2, signals)

    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label='Strategy Cumulative Returns')
    plt.title('Market Neutral Pairs Trading Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
