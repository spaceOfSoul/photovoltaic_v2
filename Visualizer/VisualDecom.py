import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
from Visualizer.SeriesDecomp import series_decomp

def visualize_decomp(dataloader, period, seasonal, show=True):
    y_values = []
    for days, (x, y) in enumerate(dataloader):
        y = y.float()
        y_values.extend(y.squeeze().tolist())

    # Calculate daily values
    daily_values = [sum(y_values[i:i+24]) for i in range(0, len(y_values), 24)]
    series = pd.Series(daily_values, dtype='float32')

    trend, seasonal, residual = series_decomp(series, period, seasonal)

    # Create an array for the x-axis (days)
    days_array = range(len(trend))

    # Plotting
    _plot_decomp(days_array, series, trend, seasonal, residual, show)


def _plot_decomp(days_array, series, trend, seasonal, residual, show):
    if show:
        fig, axarr = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

        axarr[0].plot(days_array, series, label="Original", color="blue")
        axarr[0].set_title("Original Data")
        axarr[0].set_ylabel("kW/day")
        axarr[0].set_xlabel("days")

        axarr[1].plot(days_array, trend, label="Trend")
        axarr[1].set_title("Trend")
        axarr[1].set_ylabel("kW/day")
        axarr[1].set_xlabel("days")

        axarr[2].plot(days_array, seasonal, label="Seasonal", color="orange")
        axarr[2].set_title("Seasonal")
        axarr[2].set_ylabel("kW/day")
        axarr[2].set_xlabel("days")

        axarr[3].plot(days_array, residual, label="Residual", color="green")
        axarr[3].set_title("Residual")
        axarr[3].set_ylabel("kW/day")
        axarr[3].set_xlabel("days")

        # X ticks 설정
        max_day = int(days_array[-1])  # 마지막 날짜나 일련번호를 가져온다.
        ticks = list(range(0, max_day + 1, 30))
        for ax in axarr:
            ax.set_xticks(ticks)
            
        plt.tight_layout()
        plt.show()

