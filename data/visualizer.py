import os
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from data.data_processing import DataProcessingPipeline
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

def visualize_features(columns: list[str], location_code: int = 0, no_of_days: int = 100, type_of_data: str = "processed", is_decomposed: bool = False):
    # Load data
    df = pd.read_csv(f"data/{type_of_data}-data/train.csv")
    if type_of_data == "raw":
        pipe = DataProcessingPipeline(df)
        pipe._transform_cities_to_codes()
        pipe._extract_time_series_feature_for_city()
        pipe._transform_categorical_features_to_numerical()
        df = pipe.get()
    df = df[df["Location"] == location_code]

    # Ensure necessary columns are present
    required_columns = set(columns + ["Date", "RainTomorrow", "Location"])
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by Date (integer time step) and limit to no_of_days
    df = df.sort_values("Date").head(no_of_days)

    # Min-max scale selected columns
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0.0  # Handle constant columns

    # Create plot
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Use updated colormap
    colors = plt.get_cmap("tab10", len(columns))

    # Plot features
    for idx, col in enumerate(columns):
        ax1.plot(df["Date"], df[col], label=col, color=colors(idx))

    ax1.set_xlabel("Time Step (Date)")
    ax1.set_ylabel("Feature Values")
    ax1.tick_params(axis='x', rotation=45)

    # Plot RainTomorrow as bar chart
    ax2 = ax1.twinx()
    ax2.bar(
        df["Date"], df["RainTomorrow"],
        color='orange', alpha=0.3, width=0.8,
        label="RainTomorrow (Yes=1)", zorder=1
    )
    ax2.set_ylim(0, 1.1)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Save figure
    os.makedirs("visualizations", exist_ok=True)
    feature_str = '_'.join(columns)
    filename = f"visualizations/timeseries_{type_of_data}_tmr_loc_{location_code}_{no_of_days}_{feature_str}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def visualize_decomposed_features(columns: list[str], location_code: int = 0, no_of_days: int = 100, type_of_data: str = "processed"):
    # Load data
    df = pd.read_csv(f"data/{type_of_data}-data/train.csv")
    if type_of_data == "raw":
        pipe = DataProcessingPipeline(df)
        pipe._transform_cities_to_codes()
        pipe._extract_time_series_feature_for_city()
        pipe._transform_categorical_features_to_numerical()
        df = pipe.get()
    df = df[df["Location"] == location_code]

    # Ensure necessary columns are present
    required_columns = set(columns + ["Date", "RainTomorrow", "Location"])
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by Date (integer time step) and limit to no_of_days
    df = df.sort_values("Date").head(no_of_days)
    df["Date"] = pd.to_datetime(df["Date"], unit='D', origin='unix')  # Convert to datetime

    for col in columns:
        # Min-max scale selected column
        min_val = df[col].min()
        max_val = df[col].max()
        series = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else pd.Series(0.0, index=df.index)

        # Decompose the time series
        try:
            decomposition = seasonal_decompose(series, model='additive', period=4, extrapolate_trend='freq')
        except ValueError as e:
            print(f"Skipping column {col}: {e}")
            continue

        # Plot the decomposed components
        fig, axs = plt.subplots(5, 1, figsize=(15, 10), sharex=True)

        axs[0].plot(df["Date"], series, label=f"{col} (Normalized)", color='blue')
        axs[0].set_ylabel("Observed")
        axs[0].legend(loc='upper left')

        axs[1].plot(df["Date"], decomposition.trend, label="Trend", color='green')
        axs[1].set_ylabel("Trend")
        axs[1].legend(loc='upper left')

        axs[2].plot(df["Date"], decomposition.seasonal, label="Seasonal", color='purple')
        axs[2].set_ylabel("Seasonal")
        axs[2].legend(loc='upper left')

        axs[3].plot(df["Date"], decomposition.resid, label="Residual", color='red')
        axs[3].set_ylabel("Residual")
        axs[3].legend(loc='upper left')

        # RainTomorrow as bar chart
        axs[4].bar(
            df["Date"], df["RainTomorrow"],
            color='orange', alpha=0.3, width=1.5,
            label="RainTomorrow (Yes=1)"
        )
        axs[4].set_ylabel("RainTomorrow")
        axs[4].set_ylim(0, 1.1)
        axs[4].legend(loc='upper left')
        axs[4].tick_params(axis='x', rotation=45)

        plt.suptitle(f"Decomposition of {col} (Location {location_code}, {no_of_days} days)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save figure
        os.makedirs("visualizations", exist_ok=True)
        filename = f"visualizations/decomposed_{type_of_data}_tmr_loc_{location_code}_{no_of_days}_{col}.png"
        plt.savefig(filename)
        plt.close()

def plot_correlation_heatmap(columns, type_of_data="raw"):
    df = pd.read_csv(f"data/{type_of_data}-data/train.csv")
    columns = [col for col in columns if col in df.columns]
    columns.extend(["Location", "Date", "RainToday", "RainTomorrow"])
    df = df[columns]
    corr_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(20,20))
    sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
    os.makedirs("visualizations", exist_ok=True)
    filename = f"visualizations/heatmap.png"
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    columns = [
        "MinTemp", "MaxTemp", "Rainfall",
        "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
        "Humidity9am", "Humidity3pm",
        "Pressure9am", "Pressure3pm",
        "Temp9am", "Temp3pm",
        "WindGustDir_sin", "WindGustDir_cos",
        "WindDir9am_sin", "WindDir9am_cos",
        "WindDir3pm_sin", "WindDir3pm_cos"
    ]
    for column in columns:
        plot_correlation_heatmap(columns=columns, type_of_data="processed")
        # visualize_features(columns=columns[:5], type_of_data="processed")
        # visualize_decomposed_features([column], type_of_data="processed")