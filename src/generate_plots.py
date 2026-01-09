import pandas as pd
import matplotlib.pyplot as plt
from src.config import FILTERED_CSV
from src.utils import save_plot, generate_wordcloud
import sys


def execute_plots():
    if not FILTERED_CSV.exists():
        print("Dataset not found. Please run ETL first.")
        return

    print("Loading data for plots...")
    df = pd.read_csv(FILTERED_CSV, low_memory=False)

    # 1. Histogram
    print("Generating histogram...")
    df["word_count"] = (
        df["Consumer complaint narrative"].astype(str).apply(lambda x: len(x.split()))
    )
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(df["word_count"], bins=50, color="teal", edgecolor="black")
    ax1.set_title("Distribution of Complaint Word Counts")
    ax1.set_xlabel("Words")
    ax1.set_ylabel("Frequency")
    save_plot(fig1, "word_len_dist.png")

    # 2. Bar Chart
    print("Generating bar chart...")
    top_sub = df["Sub-product"].value_counts().head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    top_sub.sort_values().plot(kind="barh", color="salmon", ax=ax2)
    ax2.set_title("Top 10 Sub-products")
    save_plot(fig2, "sub_products.png")

    # 3. Line Chart
    if "Date received" in df.columns:
        print("Generating time trend...")
        df["Date received"] = pd.to_datetime(df["Date received"])
        time_trend = df.set_index("Date received").resample("M").size()
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        time_trend.plot(ax=ax3, color="purple")
        ax3.set_title("Complaints Over Time")
        save_plot(fig3, "time_trend.png")

    # 4. WordCloud
    print("Generating word cloud...")
    generate_wordcloud(
        df["Consumer complaint narrative"].astype(str).tolist(), "wordcloud.png"
    )
    print("All plots generated.")


if __name__ == "__main__":
    execute_plots()
