import pandas as pd

# Read CSV and write with quotes
df = pd.read_csv("hashtags_classified.csv")
df.to_csv("hashtags_bq_ready.csv", index=False, quoting=1)
