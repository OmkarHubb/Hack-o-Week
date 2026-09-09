"""
data_loader.py - Loads and cleans CSV datasets using pandas.
Demonstrates: read_csv, drop_duplicates, fillna, merge
"""

import pandas as pd


def load_authentication_logs(filepath):
    """Load, deduplicate, and clean the authentication logs CSV."""
    df = pd.read_csv(filepath)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Replace missing IP addresses with "Unknown"
    df["IP_Address"] = df["IP_Address"].fillna("Unknown")
    df["IP_Address"] = df["IP_Address"].replace("", "Unknown")

    # Convert Timestamp to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    return df


def load_ip_country(filepath):
    """Load the IP-to-country mapping CSV."""
    return pd.read_csv(filepath)


def merge_with_country(logs_df, ip_country_df):
    """Left-merge authentication logs with IP-country mapping."""
    merged = pd.merge(logs_df, ip_country_df, on="IP_Address", how="left")
    merged["Country"] = merged["Country"].fillna("Unknown")
    return merged
