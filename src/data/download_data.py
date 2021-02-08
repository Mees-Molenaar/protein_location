"""
Download protein data from the uniprot database.
"""

import requests
import pandas as pd
from datetime import date
from pathlib import Path

"""
This query uses only:
Proteins that are Swiss-prot reviewed
Proteins that are found in humans
Format: Tab-separated
Columns: Entry, Entry Name, Protein Names, Gene Names, Sequence, Subcellular Location [CC]
"""
query = 'https://www.uniprot.org/uniprot/?query=*&format=tab&columns=id,entry%20name,protein%20names,genes,sequence,comment(SUBCELLULAR%20LOCATION)&fil=organism:%22Homo%20sapiens%20(Human)%20[9606]%22%20AND%20reviewed:yes'

response = requests.get(query)

if response.status_code != 200:
    raise ApiError(f'Get request failed. Status code: {response.status_code}.')

# Split the data on new lines since every entry is on a new line.
protein_data = response.text.splitlines()

# Save the columns
columns = protein_data[0].split('\t')

# Start an empty data list
data = []

# Loop over the entries, except for the first header entry. Split the entry and append it to the data list.
for entry in protein_data[1:]:
    data.append(entry.split('\t'))

# Create pandas dataframe
df = pd.DataFrame(data, columns=columns)

# Save dataframe as csv in the raw data folder
today = date.today()
file_name = f'protein_data_{today}.csv'
dir_name = '/home/mees/Desktop/Machine_Learning/subcellular_location/data/raw'

file_path = Path(dir_name, file_name) 

df.to_csv(file_path, sep=';', index=False)