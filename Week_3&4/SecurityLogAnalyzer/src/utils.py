"""
utils.py - Helper functions for the Security Log Analyzer.
Demonstrates: Functions, loops, list comprehensions, conditional statements.
"""

import os
import random
import pandas as pd
from datetime import datetime, timedelta


def ensure_directory(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def generate_sample_data(data_dir):
    """Generate sample CSV datasets if they don't already exist."""
    ensure_directory(data_dir)

    auth_path = os.path.join(data_dir, "authentication_logs.csv")
    ip_path = os.path.join(data_dir, "ip_country.csv")

    # Skip if files already exist
    if os.path.exists(auth_path) and os.path.exists(ip_path):
        return

    # --- IP to country mapping ---
    ip_country_map = {
        "192.168.1.5": "India",       "10.0.0.1": "USA",
        "172.16.0.10": "Germany",     "192.168.1.100": "India",
        "10.0.0.55": "USA",          "203.0.113.5": "China",
        "198.51.100.14": "Russia",   "192.0.2.1": "Brazil",
        "172.16.0.25": "Germany",    "10.0.0.99": "USA",
        "203.0.113.45": "China",     "198.51.100.78": "Russia",
        "192.168.1.200": "India",    "10.0.0.200": "UK",
        "172.16.0.50": "Germany",    "203.0.113.100": "China",
        "198.51.100.200": "Russia",  "192.0.2.50": "Brazil",
        "10.0.0.150": "Japan",       "172.16.0.75": "France",
    }

    ip_addresses = list(ip_country_map.keys())
    usernames = ["john", "jane", "admin", "root", "alice",
                 "bob", "charlie", "david", "eve", "frank",
                 "grace", "heidi", "ivan", "judy", "mallory"]
    devices = ["Laptop", "Desktop", "Mobile", "Tablet", "Server"]

    # --- Build ~500 authentication log records ---
    random.seed(42)
    base_time = datetime(2026, 7, 1)
    records = []

    for _ in range(500):
        timestamp = base_time + timedelta(
            days=random.randint(0, 20),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        records.append({
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Username": random.choice(usernames),
            "IP_Address": random.choice(ip_addresses),
            "Status": random.choices(["Success", "Failed"], weights=[40, 60])[0],
            "Device": random.choice(devices),
        })

    # Add ~20 duplicate rows
    records.extend([random.choice(records) for _ in range(20)])

    # Add ~15 missing IP addresses
    for idx in random.sample(range(len(records)), 15):
        records[idx]["IP_Address"] = ""

    random.shuffle(records)

    # Save CSVs
    pd.DataFrame(records).to_csv(auth_path, index=False)
    pd.DataFrame(
        list(ip_country_map.items()),
        columns=["IP_Address", "Country"],
    ).to_csv(ip_path, index=False)
