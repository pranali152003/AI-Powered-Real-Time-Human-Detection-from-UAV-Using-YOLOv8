import pandas as pd
import os

csv_file = "data/alerts.csv"

# Read alerts
df = pd.read_csv(
    csv_file,
    names=["Date", "Time", "Humans", "Level", "Snapshot"]
)

# Keep only rows whose snapshot file exists
df = df[df["Snapshot"].apply(
    lambda x: os.path.exists(os.path.join("snapshots", str(x)))
)]

# Save cleaned CSV
df.to_csv(csv_file, index=False, header=False)

# Generate Excel report
df.to_excel("data/UAV_Report.xlsx", index=False)

print("Broken entries removed and report generated.")