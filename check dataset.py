
import pandas as pd

print("Python is working!")

# HAM10000 metadata
df = pd.read_csv(r"C:\Users\dell\Downloads\HAM10000_metadata.csv")

# Disease-wise count
print("\nDisease-wise Image Count:")
print(df["dx"].value_counts())

# Total images
print("\nTotal Images:", len(df))