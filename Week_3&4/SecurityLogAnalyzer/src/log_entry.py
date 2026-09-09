"""
log_entry.py - Represents a single authentication log record.
Demonstrates: OOP, __init__, __str__, if-else
"""


class LogEntry:
    """One authentication log record."""

    def __init__(self, timestamp, username, ip_address, status):
        self.timestamp = timestamp
        self.username = username
        self.ip_address = ip_address if ip_address else "Unknown"
        self.status = status

    def is_failed(self):
        """Return True if this login attempt failed."""
        return self.status == "Failed"

    def __str__(self):
        return f"[{self.timestamp}] {self.username} from {self.ip_address} -> {self.status}"
