# Security Log Analyzer

## Project Overview

A simple command-line tool that processes server authentication logs to detect suspicious login activity and generate visual security reports.

This is a Hack-O-Week educational prototype demonstrating Python, NumPy, Pandas, and Matplotlib.

---

## Features

- Load and clean authentication log data from CSV files
- Remove duplicate records and handle missing IP addresses
- Merge logs with IP-to-country mapping
- Calculate login success and failure statistics
- Detect suspicious IPs with more than 5 failed login attempts
- Identify the top 5 most targeted usernames
- Generate 3 security charts saved as PNG

---

## Technologies Used

| Library | Purpose |
|---|---|
| Python 3 | Core language, OOP, functions |
| NumPy | Vectorized percentage calculations |
| Pandas | CSV loading, cleaning, merging, grouping |
| Matplotlib | Bar chart, horizontal bar chart, pie chart |

---

## Folder Structure

```
SecurityLogAnalyzer/
├── data/
│   ├── authentication_logs.csv
│   └── ip_country.csv
├── output/
│   ├── failed_vs_success.png
│   ├── top_suspicious_ips.png
│   └── country_distribution.png
├── src/
│   ├── __init__.py
│   ├── log_entry.py
│   ├── log_analyzer.py
│   ├── data_loader.py
│   ├── visualization.py
│   └── utils.py
├── main.py
├── requirements.txt
├── README.md
└── PROJECT_DOCUMENTATION.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

```bash
cd SecurityLogAnalyzer
python main.py
```

Sample datasets are generated automatically on first run.

---

## Example Output

```
==================================================
  Security Log Analyzer Report
==================================================

  Total Login Attempts : 500
  Successful Logins   : 203
  Failed Logins       : 297
  Success Rate        : 40.6%
  Failure Rate        : 59.4%

--------------------------------------------------
  Suspicious IPs (> 5 failed attempts)
--------------------------------------------------

  IP Address           Country      Failed Attempts
  ----------------------------------------------
  172.16.0.50          Germany      21
  203.0.113.45         China        21
  10.0.0.55            USA          20

--------------------------------------------------
  Top 5 Targeted Usernames
--------------------------------------------------

  charlie      (31 attempts)
  jane         (28 attempts)
  bob          (26 attempts)
  ivan         (25 attempts)
  root         (23 attempts)

--------------------------------------------------
  Graphs Saved
--------------------------------------------------

  > failed_vs_success.png
  > top_suspicious_ips.png
  > country_distribution.png

==================================================
```

---

## Future Scope

- Real-time log monitoring
- Dashboard using Streamlit
- Export reports to PDF
- Email alerts for suspicious activity
- Machine learning based anomaly detection
