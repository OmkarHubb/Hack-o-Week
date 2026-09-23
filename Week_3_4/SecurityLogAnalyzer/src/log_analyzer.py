"""
log_analyzer.py - Core analysis logic for the Security Log Analyzer.
Demonstrates: OOP, NumPy vectorized ops, pandas groupby/filtering,
              list/dictionary comprehensions, loops, if-else
"""

import numpy as np
import pandas as pd

from src.log_entry import LogEntry


class LogAnalyzer:
    """Analyses a merged authentication-log DataFrame."""

    SUSPICIOUS_THRESHOLD = 5  # IPs with more than this many failures

    def __init__(self, df):
        self.df = df

        # List comprehension: create LogEntry objects from rows
        self.log_entries = [
            LogEntry(
                timestamp=str(row["Timestamp"]),
                username=row["Username"],
                ip_address=row["IP_Address"],
                status=row["Status"],
            )
            for _, row in self.df.iterrows()
        ]

    def compute_login_stats(self):
        """
        Compute login statistics using NumPy vectorized operations.
        Returns a dict with total, success, failed counts and percentages.
        """
        total = len(self.df)
        failed = int((self.df["Status"] == "Failed").sum())
        success = int((self.df["Status"] == "Success").sum())

        # NumPy: vectorized percentage calculation with broadcasting
        counts = np.array([success, failed], dtype=np.float64)
        percentages = np.round((counts / total) * 100, 1)

        return {
            "total": total,
            "success": success,
            "failed": failed,
            "success_pct": percentages[0],
            "failure_pct": percentages[1],
        }

    def detect_suspicious_ips(self):
        """
        Find IPs with more than SUSPICIOUS_THRESHOLD failed attempts.
        Returns a DataFrame with IP_Address, Country, and Failed_Attempts.
        """
        failed_df = self.df[self.df["Status"] == "Failed"]

        # Group by IP and count failures
        ip_failures = failed_df.groupby("IP_Address").size()

        # Dictionary comprehension: filter suspicious IPs
        suspicious = {
            ip: count
            for ip, count in ip_failures.items()
            if count > self.SUSPICIOUS_THRESHOLD
        }

        # Build result DataFrame with country info
        rows = []
        for ip, count in suspicious.items():
            # Get country for this IP
            country_match = self.df[self.df["IP_Address"] == ip]["Country"]
            country = country_match.iloc[0] if len(country_match) > 0 else "Unknown"
            rows.append({"IP_Address": ip, "Country": country, "Failed_Attempts": count})

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("Failed_Attempts", ascending=False)

        return result

    def top_targeted_usernames(self, n=5):
        """
        Find the top-N most targeted usernames by failed attempts.
        Returns a DataFrame with Username and Failed_Attempts.
        """
        failed_df = self.df[self.df["Status"] == "Failed"]
        top = (
            failed_df["Username"]
            .value_counts()
            .head(n)
            .reset_index()
        )
        top.columns = ["Username", "Failed_Attempts"]
        return top

    def country_attack_distribution(self):
        """
        Group failed attempts by country for the pie chart.
        Returns a DataFrame with Country and Failed_Attempts.
        """
        failed_df = self.df[self.df["Status"] == "Failed"]
        distribution = (
            failed_df.groupby("Country")
            .size()
            .reset_index(name="Failed_Attempts")
            .sort_values("Failed_Attempts", ascending=False)
        )
        return distribution
