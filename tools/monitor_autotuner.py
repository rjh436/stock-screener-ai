import time
import re
from pathlib import Path

LOG_PATH = Path("/tmp/apex_autotuner.log")
INTERVAL = 60  # seconds

GEN_RE = re.compile(r"^🧬 GEN (\d+/\d+)")
WINNER_RE = re.compile(r"^🏆 WINNER: CAGR ([\-\d\.]+)% \| DD ([\-\d\.]+)% \| Calmar ([\-\d\.]+)")
AVG_VCP_RE = re.compile(r"^Avg VCP Candidates: ([\-\d\.]+)")
AVG_RS_RE = re.compile(r"^Avg RS Candidates: ([\-\d\.]+)")
AVG_TREND_RE = re.compile(r"^Avg Trend Candidates: ([\-\d\.]+)")


def tail_lines(path: Path, max_lines: int = 500):
    try:
        data = path.read_text(errors="ignore").splitlines()
    except FileNotFoundError:
        return []
    return data[-max_lines:]


def parse_snapshot(lines):
    gen = None
    winner = None
    avg_vcp = None
    avg_rs = None
    avg_trend = None

    for line in reversed(lines):
        if gen is None:
            m = GEN_RE.match(line)
            if m:
                gen = m.group(1)
                continue
        if winner is None:
            m = WINNER_RE.match(line)
            if m:
                winner = (m.group(1), m.group(2), m.group(3))
                continue
        if avg_vcp is None:
            m = AVG_VCP_RE.match(line)
            if m:
                avg_vcp = m.group(1)
                continue
        if avg_rs is None:
            m = AVG_RS_RE.match(line)
            if m:
                avg_rs = m.group(1)
                continue
        if avg_trend is None:
            m = AVG_TREND_RE.match(line)
            if m:
                avg_trend = m.group(1)
                continue
        if gen and winner and avg_vcp and avg_rs and avg_trend:
            break

    return gen, winner, avg_rs, avg_vcp, avg_trend


def print_snapshot():
    lines = tail_lines(LOG_PATH)
    gen, winner, avg_rs, avg_vcp, avg_trend = parse_snapshot(lines)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print(f"[{ts}] Autotuner Snapshot")
    print(f"Log file: {LOG_PATH}")
    print(f"Generation: {gen or 'n/a'}")
    if winner:
        cagr, dd, calmar = winner
        print(f"Best Winner: CAGR {cagr}% | DD {dd}% | Calmar {calmar}")
    else:
        print("Best Winner: n/a")
    print(f"Avg RS Candidates: {avg_rs or 'n/a'}")
    print(f"Avg VCP Candidates: {avg_vcp or 'n/a'}")
    print(f"Avg Trend Candidates: {avg_trend or 'n/a'}")


def main():
    while True:
        print_snapshot()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
