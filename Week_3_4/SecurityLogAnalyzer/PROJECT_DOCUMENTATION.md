# Security Log Analyzer - Project Documentation

## Project Objective

Develop a simple Security Log Analyzer that processes server authentication logs to identify suspicious login attempts and generate a concise security report with basic visualizations.

This is an educational prototype demonstrating Python, NumPy, Pandas, and data visualization concepts.

---

## Problem Statement

Organizations face thousands of authentication attempts daily. Without automated analysis, security teams struggle to spot brute-force attacks hidden among legitimate logins. This project provides a simple tool to detect suspicious patterns by analyzing failed login counts per IP address and visualizing attack distributions.

---

## Technologies Used

### Python
- Core programming language.
- Used for functions, object-oriented programming, loops, and conditional logic.

### Pandas
- Loading CSV files with `read_csv()`.
- Removing duplicate records with `drop_duplicates()`.
- Filling missing IP addresses with `fillna()`.
- Merging authentication logs with country data using `merge()`.
- Grouping failed attempts by IP and country with `groupby()`.
- Sorting results with `sort_values()`.
- Counting occurrences with `value_counts()`.

### NumPy
- Creating arrays from login counts.
- Vectorized percentage calculations using array broadcasting.
- Computing success and failure rates without Python loops.

### Matplotlib
- Bar chart for comparing failed vs successful logins.
- Horizontal bar chart for top suspicious IP addresses.
- Pie chart for country-wise attack distribution.
- Saving all charts as PNG files without displaying them.

---

## File Explanation

### main.py
Runs the complete analysis workflow. Loads data, performs analysis, generates charts, and prints the security report to the terminal.

### src/log_entry.py
Contains the `LogEntry` class representing a single authentication record with attributes for timestamp, username, IP address, and status.

### src/log_analyzer.py
Contains the `LogAnalyzer` class that performs all security analysis including login statistics, suspicious IP detection, top targeted usernames, and country-wise attack distribution.

### src/data_loader.py
Loads CSV files, removes duplicate rows, replaces missing IP addresses with "Unknown", and merges the authentication logs with the IP-to-country mapping.

### src/visualization.py
Creates and saves three charts: a bar chart for login outcomes, a horizontal bar chart for suspicious IPs, and a pie chart for country distribution.

### src/utils.py
Contains helper functions for creating directories and generating sample datasets when the CSV files do not exist.

---

## Python Concepts Used

### Functions
Every module uses well-named functions with a single purpose. Functions accept parameters and return results.

### Object-Oriented Programming
- `LogEntry` class represents a single log record with attributes and a helper method.
- `LogAnalyzer` class encapsulates all analysis logic with methods operating on a shared DataFrame.

### Dictionary Comprehension
Used in `detect_suspicious_ips()` to filter IPs exceeding the failure threshold:
```python
suspicious = {ip: count for ip, count in ip_failures.items() if count > 5}
```

### List Comprehension
Used to create `LogEntry` objects from DataFrame rows:
```python
self.log_entries = [LogEntry(...) for _, row in self.df.iterrows()]
```

### Loops
`for` loops iterate over DataFrames to print the report. The data generation function uses loops to build 500 sample records.

### Conditional Statements
`if-else` logic is used for checking file existence, determining login status, and handling empty results.

---

## NumPy Concepts Used

### Arrays
Login counts are converted to NumPy arrays for vectorized processing:
```python
counts = np.array([success, failed], dtype=np.float64)
```

### Vectorized Operations
Percentage calculation is performed on the entire array without loops:
```python
percentages = np.round((counts / total) * 100, 1)
```

### Broadcasting
A scalar division is broadcast across the array elements to compute success and failure rates in a single operation.

---

## Pandas Concepts Used

### Reading CSV
```python
df = pd.read_csv(filepath)
```

### Cleaning Data
```python
df = df.drop_duplicates()
df["IP_Address"] = df["IP_Address"].fillna("Unknown")
```

### Merge
```python
merged = pd.merge(logs_df, ip_country_df, on="IP_Address", how="left")
```

### GroupBy
```python
failed_df.groupby("IP_Address").size()
failed_df.groupby("Country").size()
```

### Filtering
```python
failed_df = self.df[self.df["Status"] == "Failed"]
```

### Sorting
```python
result.sort_values("Failed_Attempts", ascending=False)
```

### Value Counts
```python
failed_df["Username"].value_counts().head(5)
```

---

## Visualizations

### 1. Failed vs Successful Logins (Bar Chart)
Compares the number of successful and failed authentication attempts. A high failure count is a red flag that may indicate brute-force attacks or credential-stuffing campaigns.

### 2. Top 5 Suspicious IP Addresses (Horizontal Bar Chart)
Shows which IP addresses generated the highest number of failed login attempts. Security teams can use this to prioritize IP blocking or firewall rules.

### 3. Attack Distribution by Country (Pie Chart)
Shows the proportion of failed login attempts originating from each country. If most failures come from a region the organization does not serve, geo-blocking may be warranted.

---

## Console Output

The terminal displays a clean, concise report containing:

1. **Login Statistics** - Total attempts, successful logins, failed logins, success rate, and failure rate.
2. **Suspicious IP Table** - All IPs with more than 5 failed attempts, along with their country and failure count.
3. **Top 5 Targeted Usernames** - The five usernames that received the most failed login attempts.
4. **Graphs Saved** - Confirmation of the three chart files saved to the output folder.

---

## Future Improvements

- **Real-time log monitoring** - Stream and analyze logs as they arrive.
- **Dashboard using Streamlit** - Build an interactive web dashboard for non-technical users.
- **Export reports to PDF** - Generate downloadable security reports.
- **Email alerts** - Send automatic notifications when suspicious activity is detected.
- **Machine learning based anomaly detection** - Train models to classify login attempts as normal or malicious.
