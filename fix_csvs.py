"""
Fix CSV files: remove 'symbol' from the header line only.
The data rows never had a symbol value — the extra header was causing
all data to shift one column to the right.
"""
from pathlib import Path

data_dir = Path("data")
folders = [
    data_dir / "Income_Statments",
    data_dir / "Cash_Flow_Statments",
]

for folder in folders:
    for csv_file in folder.glob("*.csv"):
        lines = csv_file.read_text(encoding="utf-8").splitlines()
        header = lines[0]
        if ",symbol," in header:
            # Remove "symbol," from the header
            lines[0] = header.replace("symbol,", "", 1)
            csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"Fixed: {csv_file}")
        else:
            print(f"OK:    {csv_file}")
