import pandas as pd
import numpy as np
import os

def prepare_data():
    input_file = 'data/household_power_consumption.txt'
    output_file = 'data/cleaned_energy_data.csv'

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Reading {input_file}...")
    # Load data
    data = pd.read_csv(input_file, sep=';', 
                       low_memory=False, 
                       na_values=['?'])
    
    print("Processing dates...")
    # Combine Date and Time into one datetime column
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
    
    # Set Datetime as index
    data.set_index('Datetime', inplace=True)
    
    # Drop old columns
    data.drop(['Date', 'Time'], axis=1, inplace=True)
    
    print("Handling missing values...")
    # Fill missing values using forward-fill
    data.ffill(inplace=True)
    
    # Save to CSV
    print(f"Saving to {output_file}...")
    data.to_csv(output_file)
    print("Done.")

if __name__ == "__main__":
    prepare_data()
