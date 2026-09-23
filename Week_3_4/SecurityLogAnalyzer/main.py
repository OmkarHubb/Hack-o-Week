"""
main.py - Entry point for the Security Log Analyzer.
Run with: python main.py
"""

import os
import sys

# Ensure project root is on Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import generate_sample_data
from src.data_loader import load_authentication_logs, load_ip_country, merge_with_country
from src.log_analyzer import LogAnalyzer
from src.visualization import save_failed_vs_success, save_top_suspicious_ips, save_country_distribution


def main():
    # --- Paths ---
    data_dir = os.path.join(PROJECT_ROOT, "data")
    output_dir = os.path.join(PROJECT_ROOT, "output")
    auth_csv = os.path.join(data_dir, "authentication_logs.csv")
    ip_csv = os.path.join(data_dir, "ip_country.csv")

    # Step 1: Generate sample data if needed
    generate_sample_data(data_dir)

    # Step 2: Load and clean data
    logs_df = load_authentication_logs(auth_csv)
    ip_country_df = load_ip_country(ip_csv)
    merged_df = merge_with_country(logs_df, ip_country_df)

    # Step 3: Analyze
    analyzer = LogAnalyzer(merged_df)
    stats = analyzer.compute_login_stats()
    suspicious = analyzer.detect_suspicious_ips()
    top_users = analyzer.top_targeted_usernames(n=5)
    countries = analyzer.country_attack_distribution()

    # Step 4: Generate charts
    save_failed_vs_success(stats, output_dir)
    save_top_suspicious_ips(suspicious, output_dir)
    save_country_distribution(countries, output_dir)

    # --- Print Report ---
    print()
    print("=" * 50)
    print("  Security Log Analyzer Report")
    print("=" * 50)

    print(f"\n  Total Login Attempts : {stats['total']}")
    print(f"  Successful Logins   : {stats['success']}")
    print(f"  Failed Logins       : {stats['failed']}")
    print(f"  Success Rate        : {stats['success_pct']}%")
    print(f"  Failure Rate        : {stats['failure_pct']}%")

    print("\n" + "-" * 50)
    print("  Suspicious IPs (> 5 failed attempts)")
    print("-" * 50)

    if not suspicious.empty:
        print(f"\n  {'IP Address':<20} {'Country':<12} {'Failed Attempts'}")
        print("  " + "-" * 46)
        for _, row in suspicious.iterrows():
            print(f"  {row['IP_Address']:<20} {row['Country']:<12} {row['Failed_Attempts']}")
    else:
        print("\n  No suspicious IPs detected.")

    print("\n" + "-" * 50)
    print("  Top 5 Targeted Usernames")
    print("-" * 50)

    for _, row in top_users.iterrows():
        print(f"\n  {row['Username']:<12} ({row['Failed_Attempts']} attempts)")

    print("\n" + "-" * 50)
    print("  Graphs Saved")
    print("-" * 50)
    print(f"\n  > failed_vs_success.png")
    print(f"  > top_suspicious_ips.png")
    print(f"  > country_distribution.png")

    print("\n" + "=" * 50)
    print()


if __name__ == "__main__":
    main()
