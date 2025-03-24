import csv
import pandas as pd


df = pd.read_csv("data/raw-data/train.csv")
countries = sorted(df["Location"].unique())
with open("data/country_codes.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Country", "Code"])
    for i, country in enumerate(countries):
        writer.writerow([country, i])