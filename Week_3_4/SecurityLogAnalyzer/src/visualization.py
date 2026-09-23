"""
visualization.py - Generates security analysis charts as PNG files.
Demonstrates: Matplotlib bar, horizontal bar, and pie charts.
"""

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display)
import matplotlib.pyplot as plt


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_failed_vs_success(stats, output_dir):
    """Bar chart: Failed vs Successful logins."""
    ensure_dir(output_dir)

    labels = ["Successful", "Failed"]
    values = [stats["success"], stats["failed"]]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.5)

    # Annotate values on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(val), ha="center", va="bottom", fontweight="bold")

    ax.set_title("Failed vs Successful Logins", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Attempts")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "failed_vs_success.png"), dpi=150)
    plt.close(fig)


def save_top_suspicious_ips(suspicious_df, output_dir):
    """Horizontal bar chart: Top 5 suspicious IPs by failed attempts."""
    ensure_dir(output_dir)

    # Take top 5 and sort ascending (so highest bar is on top)
    plot_df = suspicious_df.head(5).sort_values("Failed_Attempts", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26"]
    ax.barh(plot_df["IP_Address"], plot_df["Failed_Attempts"],
            color=colors[-len(plot_df):], edgecolor="black")

    ax.set_title("Top Suspicious IP Addresses", fontsize=14, fontweight="bold")
    ax.set_xlabel("Failed Login Attempts")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "top_suspicious_ips.png"), dpi=150)
    plt.close(fig)


def save_country_distribution(country_df, output_dir):
    """Pie chart: Attack distribution by country."""
    ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        country_df["Failed_Attempts"],
        labels=country_df["Country"],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "black"},
    )
    ax.set_title("Attack Distribution by Country", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "country_distribution.png"), dpi=150)
    plt.close(fig)
