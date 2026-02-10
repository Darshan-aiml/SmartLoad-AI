import urllib.request
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
output_path = "data/household_power_consumption.zip"

print(f"Downloading {url} to {output_path}...")
try:
    urllib.request.urlretrieve(url, output_path)
    print("Download complete.")
except Exception as e:
    print(f"Download failed: {e}")
