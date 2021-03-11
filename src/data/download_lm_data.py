"""
Download protein sequences from the uniprot database.
"""

import requests
import pandas as pd
from datetime import date
from pathlib import Path
import time

"""
This query downloads all the protein sequences that are known from Humans.
Also the Unreviewd ones, since it is not necessary for the language model to know
with 100% certainty.

Proteins that are found in humans
Format: Tab-separated
Columns: Entry, Entry Name, Sequence
"""

limit = 1000
offset = 0

query = 'https://www.uniprot.org/uniprot/?query=*&format=tab&columns=id,entry%20name,sequence&fil=organism:%22Homo%20sapiens%20(Human)%20[9606]'


results = True
first_request = True
num_iteration = 0

while results:
    offset = num_iteration * limit
    next_query = f'{query}&limit={str(limit)}&offset={str(offset)}'
    print(f'Requesting {str(offset)} to {str(offset + limit)}.')

    response = requests.get(next_query)

    if response.status_code != 200:
        raise ApiError(f'Get request failed. Status code: {response.status_code}.')

    print('Request finished!')

    print('Splitting data to add in DF.')
    # Split the data on new lines since every entry is on a new line.
    protein_data = response.text.splitlines()

    # Save the columns
    columns = protein_data[0].split('\t')

    # Start an empty data list
    data = []

    print('Add request to DF.')
    # Loop over the entries, except for the first header entry. Split the entry and append it to the data list.
    for entry in protein_data[1:]:
        data.append(entry.split('\t'))

    # The first sequence create a new df
    if first_request:
        # Create pandas dataframe
        df = pd.DataFrame(data, columns=columns)

        # Turn of first request
        first_request = False
    else:
        # Temporarily save in a new DF and then append the dfs together
        temp_df = pd.DataFrame(data, columns=columns)
        df = df.append(temp_df, ignore_index=True)

    data_len = len(data) - 1

    # if you look human entries in the uniprot database there are ~175000 entries, so stop after 175 iterations
    # I tried to use datalen since I thought that an offset of 200000 for example would not result in anymore 
    # data. However, this is not the case

    if num_iteration > 175:
        results = False

    num_iteration += 1

    # To be nice on the server
    time.sleep(5)

# Save dataframe as csv in the raw data folder
today = date.today()
file_name = f'LM_data_{today}.csv'
dir_name = '/home/mees/Desktop/Machine_Learning/subcellular_location/data/raw'

file_path = Path(dir_name, file_name) 

df.to_csv(file_path, sep=';', index=False)
print('Successfully, saved DF as CSV file.')