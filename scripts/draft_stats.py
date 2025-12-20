#!/usr/bin/env python3
"""
Analyze draft token acceptance from log files.
Displays a 2D table of counts: rows = drafted, columns = accepted.
"""

import re
import sys

def parse_log(lines):
    """Extract (accepted, drafted) pairs from log lines."""
    pattern = r'accepted (\d+)/(\d+) draft tokens'
    pairs = []
    for line in lines:
        match = re.search(pattern, line)
        if match:
            accepted = int(match.group(1))
            drafted = int(match.group(2))
            pairs.append((accepted, drafted))
    return pairs

def build_table(pairs, max_val=16):
    """Build a 2D count table: table[drafted][accepted] = count."""
    table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
    for accepted, drafted in pairs:
        if 0 <= accepted <= max_val and 0 <= drafted <= max_val:
            table[drafted][accepted] += 1
    return table

def print_table(table):
    """Print the table in text format."""
    max_val = len(table) - 1
    
    # Find the maximum count for column width
    max_count = max(max(row) for row in table)
    cell_width = max(len(str(max_count)), 2)
    
    # Header row (accepted values)
    header = "D\\A |"
    for a in range(max_val + 1):
        header += f" {a:>{cell_width}}"
    print(header)
    print("-" * len(header))
    
    # Data rows (drafted values)
    for d in range(max_val + 1):
        row_str = f"{d:>3} |"
        for a in range(max_val + 1):
            count = table[d][a]
            if count == 0:
                row_str += f" {'·':>{cell_width}}"
            else:
                row_str += f" {count:>{cell_width}}"
        print(row_str)

def print_summary(pairs):
    """Print summary statistics."""
    if not pairs:
        print("No data found.")
        return
    
    total = len(pairs)
    total_accepted = sum(a for a, d in pairs)
    total_drafted = sum(d for a, d in pairs)
    
    print(f"\nSummary:")
    print(f"  Total records: {total}")
    print(f"  Total drafted: {total_drafted}")
    print(f"  Total accepted: {total_accepted}")
    print(f"  Overall acceptance rate: {total_accepted/total_drafted*100:.1f}%")

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()
    
    pairs = parse_log(lines)
    table = build_table(pairs)
    
    print("Draft Token Acceptance Table")
    print("Rows: Drafted (D), Columns: Accepted (A)")
    print()
    print_table(table)
    print_summary(pairs)

if __name__ == "__main__":
    main()
