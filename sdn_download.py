"""
File: sdn_download.py
Description: Downloads the SDN CSV file from the OFAC Sanctions List Service, processes it by
             renaming columns based on a custom layout, and saves records of type 'Individual'
             to a separate CSV file.
Author: Mike A.
Date: 24 Sept 2024
"""

import pandas as pd
import requests

# URL to the SDN CSV file
url = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV"

# Download SDN CSV file
response = requests.get(url)
response.raise_for_status()  # Raise an exception for unsuccessful requests

# Save file locally
with open("SDN.csv", "wb") as file:
    file.write(response.content)

# Define the custom column layout
custom_columns = [
    "ent_num", "sdn_name", "sdn_type", "program", "title", 
    "call_sign", "vess_type", "tonnage", "grt", 
    "vess_flag", "vess_owner", "remarks"
]

# Load the CSV file into Pandas 
df = pd.read_csv("SDN.csv", header=0, names=custom_columns, skiprows=1)

#df.query("sdn_type == 'Individual'").to_csv("SDN_individuals.csv", index=False)
df.to_csv("./data/SDN_individuals.csv", index=False)
print("File 'SDN_individuals.csv' has been created successfully.")

