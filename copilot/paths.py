"""Project-wide paths and constants shared by every Part."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CLEAN = os.path.join(ROOT, "data", "clean")
POLICIES = os.path.join(ROOT, "data", "policies")
DB = os.path.join(ROOT, "data", "instamart.db")
SQL = os.path.join(ROOT, "sql", "queries.sql")
OUT = os.path.join(ROOT, "outputs")
REPORTS = os.path.join(OUT, "reports")
STATE = os.path.join(ROOT, "state")

SEED = 42
TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def ensure_dirs():
    for d in (CLEAN, OUT, REPORTS, STATE):
        os.makedirs(d, exist_ok=True)


class Report:
    """Prints a line and also keeps it, so each script can save its report under outputs/reports/."""

    def __init__(self, name):
        self.name, self.lines = name, []

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line)
        self.lines.append(line)

    def save(self):
        ensure_dirs()
        with open(os.path.join(REPORTS, self.name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
